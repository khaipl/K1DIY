#include "booster_vision/vision_node.h"

#include <cstdlib>
#include <functional>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <chrono>

#include <opencv2/imgproc.hpp> 
#include <opencv2/highgui.hpp> // Required for cv::imshow
#include <yaml-cpp/yaml.h>
#include <cv_bridge/cv_bridge.h>
#include "ament_index_cpp/get_package_share_directory.hpp"

// [LIB] Internal Modules: Kept strictly to synchronization and image translation
#include "booster_vision/base/data_syncer.hpp"     // Matches Color + Depth images by time
#include "booster_vision/base/data_logger.hpp"     // Logs raw data for offline training
#include "booster_vision/base/misc_utils.hpp"
#include "booster_vision/img_bridge.h"             // Converts ROS images to OpenCV
#include "booster_vision/model/detector.h"
#include "booster_vision/model/segmentor.h"        // Added for Field Lines
#include "booster_vision/pose_estimator/pose_estimator.h"

// [LIB] Custom ROS 2 Messages for the Brain Node
#include "vision_interface/msg/detected_object.hpp"
#include "vision_interface/msg/detections.hpp"
#include "vision_interface/msg/line_segments.hpp"  // Added for Brain Localization

namespace booster_vision {
// =================================================================================================
// [CONSTRUCTOR]
// Role: Creates the ROS 2 node with the given name.
// =================================================================================================
VisionNode::VisionNode(const std::string &node_name, const rclcpp::NodeOptions &options) 
    : Node(node_name, options) {
    this->declare_parameter<bool>("offline_mode", false);
    this->declare_parameter<bool>("show_det", false);
    this->declare_parameter<bool>("show_seg", false);
    this->declare_parameter<bool>("save_data", false);
    this->declare_parameter<bool>("save_depth", false);
    this->declare_parameter<int>("save_fps", 3);
    this->declare_parameter<std::string>("robot_name", "");
    this->declare_parameter<std::string>("detection_model_path", "");
    this->declare_parameter<std::string>("segmentation_model_path", "");
    this->declare_parameter<std::string>("color_topic", "");
    this->declare_parameter<std::string>("depth_topic", "");
    this->declare_parameter<std::string>("intrin_topic", "");
}

// =================================================================================================
// [FUNCTION] Init
// Role: The Setup Phase. Loads YAML configs and initializes DataSyncer, AI, and ROS Subs.
// =================================================================================================
void VisionNode::Init(const std::string &cfg_template_path, const std::string &cfg_path) {
    
    // --- 1. Load Configuration ---
    if (!std::filesystem::exists(cfg_template_path)) {
        std::cerr << "Error: Configuration template file '" << cfg_template_path << "' does not exist." << std::endl;
        return;
    }

    // Load into a temporary root node first
    YAML::Node node = YAML::LoadFile(cfg_template_path);
    if (!std::filesystem::exists(cfg_path)) {
        std::cout << "Warning: Configuration file empty!" << std::endl;
    } else {
        YAML::Node cfg_node = YAML::LoadFile(cfg_path);
        MergeYAML(node, cfg_node);
    }
    
    config_node_ = node;
    std::cout << "loaded file: " << std::endl << node << std::endl;

    // --- Retrieve Parameters ---
    this->get_parameter<bool>("show_det", show_det_);
    this->get_parameter<bool>("show_seg", show_seg_);
    this->get_parameter<bool>("save_data", save_data_);
    this->get_parameter<bool>("save_depth", save_depth_);
    this->get_parameter<bool>("offline_mode", offline_mode_);
    this->get_parameter<std::string>("color_topic", color_topic_);
    this->get_parameter<std::string>("depth_topic", depth_topic_);
    this->get_parameter<std::string>("intrin_topic", intrin_topic_);
    this->get_parameter<std::string>("detection_model_path", detection_model_path);
    this->get_parameter<std::string>("segmentation_model_path", segmentation_model_path);
    
    int save_fps = 0;
    this->get_parameter<int>("save_fps", save_fps);
    save_depth_ = save_depth_ && save_data_;
    save_every_n_frame_ = std::max(1, save_fps > 0 ? 30 / save_fps : 1);

    // K1DIY custom params from YAML fallback
    is_recording_ = as_or<bool>(node["is_recording"], false);
    show_det_ = as_or<bool>(node["show_det"], show_det_);
    show_seg_ = as_or<bool>(node["show_seg"], show_seg_);

    std::string robot_name = as_or<std::string>(node["robot_name"], "");
    std::string topic_suffix = robot_name.empty() ? "" : "/" + robot_name;
    robot_name_ = robot_name;

    // --- 2. Load Camera Mathematics (Calibration) ---
    if (!node["camera"]) {
        std::cerr << "no camera param found here" << std::endl;
        return;
    } else {
        if (color_topic_.empty()) color_topic_ = node["camera"]["camera_topic"].as<std::string>();
        if (depth_topic_.empty()) depth_topic_ = node["camera"]["depth_topic"].as<std::string>();
        if (intrin_topic_.empty()) intrin_topic_ = node["camera"]["intrin_topic"].as<std::string>();

        camera_type_ = as_or<std::string>(node["camera"]["type"], "realsense");
        intr_ = Intrinsics(node["camera"]["intrin"]);
        p_eye2head_ = as_or<Pose>(node["camera"]["extrin"], Pose());

        float pitch_comp = as_or<float>(node["camera"]["pitch_compensation"], 0.0);
        float yaw_comp = as_or<float>(node["camera"]["yaw_compensation"], 0.0);
        float z_comp = as_or<float>(node["camera"]["z_compensation"], 0.0);
        p_headprime2head_ = Pose(0, 0, z_comp, 0, pitch_comp * M_PI / 180, yaw_comp * M_PI / 180);
    }

    // --- 3. Initialize YOLO Detector (RESTORED K1DIY BACKEND LOGIC) ---
    if (node["detection_model"]) {
        std::string backend = as_or<std::string>(node["detection_model"]["backend"], "tensorrt");
        std::string active_model_path;

        if (backend == "cpu_onnx") {
            active_model_path = as_or<std::string>(node["detection_model"]["model_path_onnx"], "");
            std::cout << "Configuring for ONNX Backend..." << std::endl;
        } else {
            active_model_path = as_or<std::string>(node["detection_model"]["model_path_tensorrt"], "");
            std::cout << "Configuring for TensorRT Backend..." << std::endl;
        }

        // Safely resolve relative paths (e.g., "./src/...") to the absolute install directory
        if(!active_model_path.empty() && active_model_path[0] != '/') {
            std::string package_path = ament_index_cpp::get_package_share_directory("vision");
            active_model_path = (std::filesystem::path(package_path) / active_model_path).string();
        }

        detector_ = YoloV8Detector::CreateYoloV8Detector(node["detection_model"], active_model_path);
        detection_model_path = active_model_path; // Save for error logging
        classnames_ = node["detection_model"]["classnames"].as<std::vector<std::string>>();
        
        float default_threshold = as_or<float>(node["detection_model"]["confidence_threshold"], 0.2);
        if (node["detection_model"]["post_process"]) {
            enable_post_process_ = true;
            single_ball_assumption_ = as_or<bool>(node["detection_model"]["post_process"]["single_ball_assumption"], false);
            if (node["detection_model"]["post_process"]["confidence_thresholds"]) {
                for (const auto &item : node["detection_model"]["post_process"]["confidence_thresholds"]) {
                    confidence_map_[item.first.as<std::string>()] = item.second.as<float>();
                }
                for (const auto &classname : classnames_) {
                    if (confidence_map_.find(classname) == confidence_map_.end()) {
                        confidence_map_[classname] = default_threshold;
                    }
                }
            }
        }
    }

    // --- 4. Initialize YOLO Segmentor (RESTORED K1DIY BACKEND LOGIC) ---
    if (node["segmentation_model"]) {
        std::string backend = as_or<std::string>(node["segmentation_model"]["backend"], "tensorrt");
        std::string active_seg_model_path;

        if (backend == "cpu_onnx") {
            active_seg_model_path = as_or<std::string>(node["segmentation_model"]["model_path_onnx"], "");
        } else {
            active_seg_model_path = as_or<std::string>(node["segmentation_model"]["model_path_tensorrt"], "");
        }

        // Safely resolve relative paths for the segmentor
        if(!active_seg_model_path.empty() && active_seg_model_path[0] != '/') {
            std::string package_path = ament_index_cpp::get_package_share_directory("vision");
            active_seg_model_path = (std::filesystem::path(package_path) / active_seg_model_path).string();
        }

        segmentor_ = YoloV8Segmentor::CreateYoloV8Segmentor(node["segmentation_model"], active_seg_model_path);
    }

    // --- 5. Initialize Color Classifier ---
    if (node["robot_color_classifier"]) {
        color_classifier_ = std::make_shared<ColorClassifier>();
        color_classifier_->Init(node["robot_color_classifier"]);
    }

    // --- 6. Initialize Pose Estimators (2D -> 3D Projection) ---
    pose_estimator_ = std::make_shared<PoseEstimator>(intr_);
    pose_estimator_->Init(YAML::Node());
    pose_estimator_map_["default"] = pose_estimator_;

    if (node["ball_pose_estimator"]) {
        pose_estimator_map_["ball"] = std::make_shared<BallPoseEstimator>(intr_);
        pose_estimator_map_["ball"]->Init(node["ball_pose_estimator"]);
    }
    if (node["human_like_pose_estimator"]) {
        pose_estimator_map_["human_like"] = std::make_shared<HumanLikePoseEstimator>(intr_);
        pose_estimator_map_["human_like"]->Init(node["human_like_pose_estimator"]);
    }
    if (node["field_marker_pose_estimator"]) {
        pose_estimator_map_["field_marker"] = std::make_shared<FieldMarkerPoseEstimator>(intr_);
        pose_estimator_map_["field_marker"]->Init(node["field_marker_pose_estimator"]);
        line_segment_area_threshold_ = as_or<int>(node["field_marker_pose_estimator"]["line_segment_area_threshold"], 75);
    }

    // --- 7. Setup Synchronization & Logging ---
    use_depth_ = as_or<bool>(node["use_depth"], false);
    data_syncer_ = std::make_shared<DataSyncer>(use_depth_);
    seg_data_syncer_ = std::make_shared<DataSyncer>(false); // 2D only, no depth needed for lines

    bool save_data_nonstationary = as_or<bool>(node["misc"]["save_data_nonstationary"], true);
    std::string log_root = std::string(std::getenv("HOME")) + "/Workspace/vision_log/" + getTimeString();
    data_logger_ = save_data_ ? std::make_shared<DataLogger>(log_root, save_data_nonstationary) : nullptr;
    if (data_logger_) data_logger_->LogYAML(node, "vision_local.yaml");

    // --- 8. ROS 2 Communication Setup (with Multithreading) ---
    // Fix Topic Names for Simulation
    if (color_topic_.find("robot0_rgbd_camera") != std::string::npos && !robot_name.empty()) {
        color_topic_ = color_topic_.replace(color_topic_.find("robot0_rgbd_camera"), 18, robot_name + "_rgbd_camera");
    }
    if (depth_topic_.find("robot0_rgbd_camera") != std::string::npos && !robot_name.empty()) {
        depth_topic_ = depth_topic_.replace(depth_topic_.find("robot0_rgbd_camera"), 18, robot_name + "_rgbd_camera");
    }
    if (intrin_topic_.find("robot0_rgbd_camera") != std::string::npos && !robot_name.empty()) {
        intrin_topic_ = intrin_topic_.replace(intrin_topic_.find("robot0_rgbd_camera"), 18, robot_name + "_rgbd_camera");
    }

    // Callback Groups for performance
    callback_group_sub_1_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    callback_group_sub_2_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    callback_group_sub_3_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    callback_group_sub_4_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    
    auto sub_opt_1 = rclcpp::SubscriptionOptions(); sub_opt_1.callback_group = callback_group_sub_1_;
    auto sub_opt_2 = rclcpp::SubscriptionOptions(); sub_opt_2.callback_group = callback_group_sub_2_;
    auto sub_opt_3 = rclcpp::SubscriptionOptions(); sub_opt_3.callback_group = callback_group_sub_3_;
    auto sub_opt_4 = rclcpp::SubscriptionOptions(); sub_opt_4.callback_group = callback_group_sub_4_;

    it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());

    // Subscriptions
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        intrin_topic_, rclcpp::QoS(1).best_effort(),
        std::bind(&VisionNode::CameraInfoCallback, this, std::placeholders::_1));

    if (color_topic_.find("compressed") != std::string::npos) {
        compressed_color_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            color_topic_, 1, std::bind(&VisionNode::CompressedColorCallback, this, std::placeholders::_1), sub_opt_1);
    } else {
        color_sub_ = it_->subscribe(color_topic_, 1, &VisionNode::ColorCallback, this, nullptr, sub_opt_1);
    }

    if (use_depth_ && depth_topic_.find("compressed") != std::string::npos) {
        compressed_depth_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            depth_topic_, 1, std::bind(&VisionNode::CompressedDepthCallback, this, std::placeholders::_1), sub_opt_3);
    } else if (use_depth_) {
        depth_sub_ = it_->subscribe(depth_topic_, 1, &VisionNode::DepthCallback, this, nullptr, sub_opt_3);
    }

    if (offline_mode_) {
        pose_tf_sub_ = this->create_subscription<geometry_msgs::msg::TransformStamped>(
            "/booster_soccer/t_head2base" + topic_suffix, 10, std::bind(&VisionNode::PoseTFCallBack, this, std::placeholders::_1));
    } else {
        pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
            "/head_pose" + topic_suffix, 10, std::bind(&VisionNode::PoseCallBack, this, std::placeholders::_1), sub_opt_4);
        calParam_sub_ = this->create_subscription<vision_interface::msg::CalParam>(
            "/booster_soccer/cal_param" + topic_suffix, 10, std::bind(&VisionNode::CalParamCallback, this, std::placeholders::_1));
        pose_tf_pub_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("/booster_soccer/t_head2base" + topic_suffix, rclcpp::QoS(10));
    }

    // Publishers
    detection_pub_ = this->create_publisher<vision_interface::msg::Detections>("/booster_soccer/detection" + topic_suffix, rclcpp::QoS(1));
    ball_pub_ = this->create_publisher<vision_interface::msg::Ball>("/booster_soccer/ball" + topic_suffix, rclcpp::QoS(1));

    if (segmentor_) {
        if (color_topic_.find("compressed") != std::string::npos) {
            compressed_color_seg_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
                color_topic_, 1, std::bind(&VisionNode::CompressedSegmentationCallback, this, std::placeholders::_1), sub_opt_2);
        } else {
            color_seg_sub_ = it_->subscribe(color_topic_, 1, &VisionNode::SegmentationCallback, this, nullptr, sub_opt_2);
        }
        field_line_pub_ = this->create_publisher<vision_interface::msg::LineSegments>("/booster_soccer/line_segments" + topic_suffix, rclcpp::QoS(1));
    }

    // Debug View Publishers
    if (show_det_) detection_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/booster_soccer/debug_det_img", rclcpp::QoS(1));
    if (show_seg_) segmentation_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/booster_soccer/debug_seg_img", rclcpp::QoS(1));
}

// =================================================================================================
// [FUNCTION] ProcessData
// Role: Runs AI inference, projects 2D pixels to 3D field coordinates, and publishes Detections.
// =================================================================================================
void VisionNode::ProcessData(SyncedDataBlock &synced_data, vision_interface::msg::Detections &detection_msg) {
    if (!detector_) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
            "FATAL: detector_ is NULL! The ONNX/TensorRT model failed to load. Check path: %s", 
            detection_model_path.c_str());
        return;
    }

    double timestamp = synced_data.color_data.timestamp;
    double depth_time_diff = (timestamp - synced_data.depth_data.timestamp) * 1000;
    double pose_time_diff = (timestamp - synced_data.pose_data.timestamp) * 1000;
    if (use_depth_ && depth_time_diff > 40) std::cerr << "color depth time diff: " << depth_time_diff << "ms" << std::endl;
    if (pose_time_diff > 40) std::cerr << "color pose time diff: " << pose_time_diff << " ms" << std::endl;

    cv::Mat color = synced_data.color_data.data;
    cv::Mat depth = synced_data.depth_data.data;

    // Convert depth for the pose estimator
    cv::Mat depth_float;
    if (!depth.empty() && depth.depth() == CV_16U) {
        depth.convertTo(depth_float, CV_32F, 0.001, 0); 
    } else {
        depth_float = depth;
    }

    // Kinematic chain: Head to Base -> Eye to Base
    Pose p_head2base = synced_data.pose_data.data;
    
    // [K1DIY] Laptop Debugging Override: Provide a mock height so 2D->3D projection doesn't fail
    if (camera_type_ == "laptop") {
        p_head2base = Pose(0.0, 0.0, 0.5, 0.0, 0.2, 0.0);
    }

    Pose p_eye2base = p_head2base * p_headprime2head_ * p_eye2head_;

    // 1. AI Inference
    auto detections = detector_->Inference(color);

    auto get_estimator = [&](const std::string &class_name) {
        if (class_name == "Ball") {
            return pose_estimator_map_.count("ball") ? pose_estimator_map_["ball"] : pose_estimator_map_["default"];
        } else if (class_name == "Person" || class_name == "Opponent" || class_name == "Goalpost") {
            return pose_estimator_map_.count("human_like") ? pose_estimator_map_["human_like"] : pose_estimator_map_["default"];
        } else if (class_name.find("Cross") != std::string::npos || class_name == "PenaltyPoint") {
            return pose_estimator_map_.count("field_marker") ? pose_estimator_map_["field_marker"] : pose_estimator_map_["default"];
        }
        return pose_estimator_map_["default"];
    };

    // 2. Post-processing
    std::vector<booster_vision::DetectionRes> filtered_detections;
    if (enable_post_process_ && !detections.empty()) {
        for (auto &det : detections) {
            if (confidence_map_.empty() || det.confidence >= confidence_map_[classnames_[det.class_id]]) {
                filtered_detections.push_back(det);
            }
        }
        
        if (single_ball_assumption_) {
            std::vector<booster_vision::DetectionRes> ball_dets, other_dets;
            for (const auto &det : filtered_detections) {
                (classnames_[det.class_id] == "Ball" ? ball_dets : other_dets).push_back(det);
            }
            filtered_detections = other_dets;
            if (!ball_dets.empty()) {
                auto max_ball = *std::max_element(ball_dets.begin(), ball_dets.end(),
                    [](const auto &a, const auto &b) { return a.confidence < b.confidence; });
                filtered_detections.push_back(max_ball);
            }
        }
    } else {
        filtered_detections = detections;
    }

    // 3. 2D to 3D Spatial Projection
    for (auto &detection : filtered_detections) {
        vision_interface::msg::DetectedObject detection_obj;
        detection.class_name = detector_->kClassLabels[detection.class_id];

        auto pose_estimator = get_estimator(detection.class_name);
        
        Pose pose_obj_by_color = pose_estimator->EstimateByColor(p_eye2base, detection, color);
        Pose pose_obj_by_depth = pose_estimator->EstimateByDepth(p_eye2base, detection, color, depth_float);

        if (pose_estimator->use_depth_ && detection.class_name == "Ball" && pose_obj_by_depth == Pose()) continue;

        detection_obj.position_projection = pose_obj_by_color.getTranslationVec();
        detection_obj.position = pose_obj_by_depth.getTranslationVec();

        auto xyz = p_head2base.getTranslationVec();
        auto rpy = p_head2base.getEulerAnglesVec();
        detection_obj.received_pos = {xyz[0], xyz[1], xyz[2],
                                      static_cast<float>(rpy[0] / CV_PI * 180), 
                                      static_cast<float>(rpy[1] / CV_PI * 180), 
                                      static_cast<float>(rpy[2] / CV_PI * 180)};

        detection_obj.confidence = detection.confidence * 100;
        detection_obj.xmin = detection.bbox.x;
        detection_obj.ymin = detection.bbox.y;
        detection_obj.xmax = detection.bbox.x + detection.bbox.width;
        detection_obj.ymax = detection.bbox.y + detection.bbox.height;
        detection_obj.label = detection.class_name;

        if (color_classifier_ != nullptr && detection.class_name == "Opponent") {
            cv::Mat crop = color(detection.bbox);
            detection_obj.color = color_classifier_->Classify(crop);
        }

        detection_msg.detected_objects.push_back(detection_obj);
    }

    // 4. Compute Image Corner Points Position (For Locomotion/Brain field of view)
    std::vector<cv::Point2f> corner_uvs = {
        cv::Point2f(0, 0), cv::Point2f(color.cols - 1, 0),
        cv::Point2f(color.cols - 1, color.rows - 1), cv::Point2f(0, color.rows - 1),
        cv::Point2f(color.cols / 2.0, color.rows / 2.0)
    };
    for (auto &uv : corner_uvs) {
        auto corner_pos = CalculatePositionByIntersection(p_eye2base, uv, intr_);
        detection_msg.corner_pos.push_back(corner_pos.x);
        detection_msg.corner_pos.push_back(corner_pos.y);
    }

    // ==========================================================
    // --- [K1DIY FEATURE] DEBUG VISUALIZER & VIDEO RECORDER ---
    // ==========================================================
    if (show_det_ || is_recording_) {
        cv::Mat det_img_out = YoloV8Detector::DrawDetection(color, filtered_detections);
        
        if (show_det_ && detection_img_pub_) {
            std_msgs::msg::Header header; header.stamp = this->get_clock()->now();
            sensor_msgs::msg::Image::SharedPtr debug_msg = cv_bridge::CvImage(header, "bgr8", det_img_out).toImageMsg();
            detection_img_pub_->publish(*debug_msg);
        }
        
        if (is_recording_) {
            RecordDebugVideo(raw_writer_, det_img_out, "det_video.avi", "DETECTIONS", 10.0);
        }
    } else if (raw_writer_.isOpened()) {
        raw_writer_.release();
    }

    // 5. Publish to ROS 2 Topic & Logger
    detection_pub_->publish(detection_msg);

    {
        static double last_pub_time = -1.0;
        static uint64_t count = 0;
        double pub_ts = detection_msg.header.stamp.sec +
                        static_cast<double>(detection_msg.header.stamp.nanosec) * 1e-9;
        if (last_pub_time >= 0.0) {
            double diff_ms = (pub_ts - last_pub_time) * 1000.0;
            std::cout << "[Detections Pub Interval] #" << (count) 
                      << " -> #" << (count + 1) << ": " << diff_ms << " ms" << std::endl;
        }
        last_pub_time = pub_ts;
        count++;
    }


    if (save_data_ && data_logger_) {
        save_cnt_++;
        if (save_cnt_ % save_every_n_frame_ == 0) {
            data_logger_->LogDataBlock(synced_data);
            save_cnt_ = 0;
        }
    }
}

// =================================================================================================
// [FUNCTION] ProcessSegmentationData
// Role: Runs Seg inference and fits field line segments for the locator (Brain node).
// =================================================================================================
void VisionNode::ProcessSegmentationData(SyncedDataBlock &synced_data, vision_interface::msg::LineSegments &field_line_segs_msg) {
    if (!segmentor_) return;

    cv::Mat color = synced_data.color_data.data;
    Pose p_head2base = synced_data.pose_data.data;

    // [K1DIY] Laptop Debugging Override
    if (camera_type_ == "laptop") {
        p_head2base = Pose(0.0, 0.0, 0.5, 0.0, 0.2, 0.0);
    }

    Pose p_eye2base = p_head2base * p_headprime2head_ * p_eye2head_;

    // 1. AI Inference
    auto segmentations = segmentor_->Inference(color);
    std::vector<FieldLineSegment> field_line_segs;

    // 2. Fit Contours to 3D Field Lines
    for (auto &seg : segmentations) {
        auto line_segs = FitFieldLineSegments(p_eye2base, intr_, seg.contour, line_segment_area_threshold_);
        
        for (auto line_seg : line_segs) {
            float inlier_precentage = static_cast<float>(line_seg.inlier_count) / line_seg.contour_2d_points.size();
            if (inlier_precentage < 0.25) continue;

            // 3D Coordinates
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[0].x);
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[0].y);
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[1].x);
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[1].y);

            // 2D UV Coordinates
            field_line_segs_msg.coordinates_uv.push_back(line_seg.end_points_2d[0].x);
            field_line_segs_msg.coordinates_uv.push_back(line_seg.end_points_2d[0].y);
            field_line_segs_msg.coordinates_uv.push_back(line_seg.end_points_2d[1].x);
            field_line_segs_msg.coordinates_uv.push_back(line_seg.end_points_2d[1].y);

            field_line_segs.push_back(line_seg);
        }
    }

    // ==========================================================
    // --- [K1DIY FEATURE] DEBUG VISUALIZER & VIDEO RECORDER ---
    // ==========================================================
    if (show_seg_ || is_recording_) {
        cv::Mat seg_img_out = YoloV8Segmentor::DrawSegmentation(color, segmentations);
        seg_img_out = DrawFieldLineSegments(seg_img_out, field_line_segs);
        cv::cvtColor(seg_img_out, seg_img_out, cv::COLOR_RGB2BGR);
        
        if (show_seg_ && segmentation_img_pub_) {
            std_msgs::msg::Header header; header.stamp = this->get_clock()->now();
            sensor_msgs::msg::Image::SharedPtr debug_msg = cv_bridge::CvImage(header, "bgr8", seg_img_out).toImageMsg();
            segmentation_img_pub_->publish(*debug_msg);
        }

        if (is_recording_) {
            RecordDebugVideo(depth_writer_, seg_img_out, "seg_video.avi", "SEGMENTATION", 6.0);
        }
    } else if (depth_writer_.isOpened()) {
        depth_writer_.release();
    }

    // 3. Publish
    field_line_pub_->publish(field_line_segs_msg);
}

// =================================================================================================
// Standard Callbacks
// =================================================================================================
void VisionNode::ColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    if (!msg) return;
    cv::Mat img;
    try { img = toCVMat(*msg); } catch (std::exception &e) { return; }
    if (msg->encoding == "rgb8") cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    
    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    SyncedDataBlock synced_data;

    // [K1DIY FIX] Bypass the strict DataSyncer for laptop testing
    if (camera_type_ == "laptop") {
        synced_data.color_data = ColorDataBlock(img, timestamp);
        // Force a mock pose so projection mathematics don't crash
        synced_data.pose_data = PoseDataBlock(Pose(0.0, 0.0, 0.5, 0.0, 0.2, 0.0), timestamp); 

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
            "Camera frame received! Bypassing DataSyncer and sending to AI...");
    } else {
        synced_data = data_syncer_->getSyncedDataBlock(ColorDataBlock(img, timestamp));
    }

    if (synced_data.color_data.data.empty()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
            "DataSyncer dropped frame! Check time sync or missing /head_pose and Depth.");
        return;
    }

    vision_interface::msg::Detections detection_msg;
    detection_msg.header = msg->header;
    ProcessData(synced_data, detection_msg);
}

void VisionNode::SegmentationCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    if (!msg || !segmentor_) return;
    cv::Mat img;
    try { img = toCVMat(*msg).clone(); } catch (std::exception &e) { return; }
    if (msg->encoding == "rgb8") cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    SyncedDataBlock synced_data = seg_data_syncer_->getSyncedDataBlock(ColorDataBlock(img, timestamp));
    if (synced_data.color_data.data.empty()) return;

    vision_interface::msg::LineSegments field_line_segs_msg;
    field_line_segs_msg.header = msg->header;
    ProcessSegmentationData(synced_data, field_line_segs_msg);
}

void VisionNode::DepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    cv::Mat img;
    try { img = toCVMat(*msg); } catch (std::exception &e) { return; }
    if (img.empty() || (img.depth() != CV_16U && img.depth() != CV_32F)) return;

    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    data_syncer_->AddDepth(DepthDataBlock(img, timestamp));
}

void VisionNode::PoseCallBack(const geometry_msgs::msg::Pose::SharedPtr msg) {
    auto current_time = this->get_clock()->now();
    double timestamp = static_cast<double>(current_time.nanoseconds()) * 1e-9;

    auto pose = Pose(msg->position.x, msg->position.y, msg->position.z, 
                     msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
    data_syncer_->AddPose(PoseDataBlock(pose, timestamp));
    if (seg_data_syncer_) seg_data_syncer_->AddPose(PoseDataBlock(pose, timestamp));

    if (!offline_mode_) {
        auto tf_msg = pose.toRosTFMsg();
        tf_msg.header.stamp = builtin_interfaces::msg::Time(current_time);
        tf_msg.header.frame_id = "odom";
        tf_msg.child_frame_id = "head_pose";
        pose_tf_pub_->publish(tf_msg);
    }
}

// =================================================================================================
// Restored Missing Callbacks (Compressed Streams, offline mode, and calibration)
// =================================================================================================
void VisionNode::CompressedColorCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    if (!msg) return;
    cv::Mat img;
    try {
        img = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
        if (img.empty()) return;
    } catch (std::exception &e) { return; }

    vision_interface::msg::Detections detection_msg;
    detection_msg.header = msg->header;
    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    SyncedDataBlock synced_data = data_syncer_->getSyncedDataBlock(ColorDataBlock(img, timestamp));
    if (synced_data.color_data.data.empty()) return;
    ProcessData(synced_data, detection_msg);
}

void VisionNode::CompressedSegmentationCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    if (!msg || !segmentor_) return;
    cv::Mat img;
    try {
        img = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
        if (img.empty()) return;
    } catch (std::exception &e) { return; }

    vision_interface::msg::LineSegments field_line_segs_msg;
    field_line_segs_msg.header = msg->header;
    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    SyncedDataBlock synced_data = seg_data_syncer_->getSyncedDataBlock(ColorDataBlock(img, timestamp));
    if (synced_data.color_data.data.empty()) return;
    ProcessSegmentationData(synced_data, field_line_segs_msg);
}

void VisionNode::CompressedDepthCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    cv::Mat img;
    try {
        cv::Mat compressed_data = cv::Mat(msg->data);
        img = cv::imdecode(compressed_data, cv::IMREAD_ANYDEPTH);
        if (img.empty() || (img.depth() != CV_16U && img.depth() != CV_32F)) return;
    } catch (std::exception &e) { return; }

    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    data_syncer_->AddDepth(DepthDataBlock(img, timestamp));
}

void VisionNode::PoseTFCallBack(const geometry_msgs::msg::TransformStamped::SharedPtr msg) {
    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    data_syncer_->AddPose(PoseDataBlock(Pose(*msg), timestamp));
    if(seg_data_syncer_) seg_data_syncer_->AddPose(PoseDataBlock(Pose(*msg), timestamp));
}

void VisionNode::CalParamCallback(const vision_interface::msg::CalParam::SharedPtr msg) {
    float pitch_comp = msg->pitch_compensation;
    float yaw_comp = msg->yaw_compensation;
    float z_comp = msg->z_compensation;
    p_headprime2head_ = Pose(0, 0, z_comp, 0, pitch_comp * M_PI / 180, yaw_comp * M_PI / 180);
}

void VisionNode::CameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    if (!msg) return;
    static bool first_update = true;
    float fx = msg->k[0], fy = msg->k[4], cx = msg->k[2], cy = msg->k[5];
    std::vector<float> distortion_coeffs(msg->d.begin(), msg->d.end());

    Intrinsics::DistortionModel distortion_model = Intrinsics::DistortionModel::kNone;
    float distortion_sum = 0.0;
    for (auto coeff : distortion_coeffs) distortion_sum += std::abs(coeff);

    if (distortion_sum >= 1e-6 && !distortion_coeffs.empty()) {
        distortion_model = (msg->distortion_model == "plumb_bob") ? 
            Intrinsics::DistortionModel::kInverseBrownConrady : Intrinsics::DistortionModel::kBrownConrady;
    }
    
    Intrinsics new_intr(fx, fy, cx, cy, distortion_coeffs, distortion_model);
    
    bool should_update = first_update || std::abs(intr_.fx - fx) > 0.1 || std::abs(intr_.fy - fy) > 0.1 || 
                         std::abs(intr_.cx - cx) > 0.1 || std::abs(intr_.cy - cy) > 0.1;
    
    if (should_update) {
        intr_ = new_intr;
        pose_estimator_map_.clear();
        pose_estimator_map_["default"] = std::make_shared<PoseEstimator>(intr_);
        if (config_node_["ball_pose_estimator"]) {
            pose_estimator_map_["ball"] = std::make_shared<BallPoseEstimator>(intr_);
            pose_estimator_map_["ball"]->Init(config_node_["ball_pose_estimator"]);
        }
        if (config_node_["human_like_pose_estimator"]) {
            pose_estimator_map_["human_like"] = std::make_shared<HumanLikePoseEstimator>(intr_);
            pose_estimator_map_["human_like"]->Init(config_node_["human_like_pose_estimator"]);
        }
        if (config_node_["field_marker_pose_estimator"]) {
            pose_estimator_map_["field_marker"] = std::make_shared<FieldMarkerPoseEstimator>(intr_);
            pose_estimator_map_["field_marker"]->Init(config_node_["field_marker_pose_estimator"]);
        }
        first_update = false;
    }
    
    static int received_count = 0;
    if (++received_count >= 5) camera_info_sub_.reset();
}

// =================================================================================================
// [FUNCTION] RecordDebugVideo
// =================================================================================================
void VisionNode::RecordDebugVideo(cv::VideoWriter& writer, const cv::Mat& frame, const std::string& filename, const std::string& log_name, double fps) {
    if (frame.empty()) return;
    std::string save_path = "data/test/";

    if (!writer.isOpened()) {
        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        std::filesystem::create_directories(save_path);
        writer.open(save_path + filename, codec, fps, frame.size(), true);
        if (start_record_time_.nanoseconds() == 0) start_record_time_ = this->get_clock()->now();
    }

    auto elapsed = this->get_clock()->now() - start_record_time_;
    
    if (elapsed.seconds() <= 10.0) {
        writer.write(frame);
    } else {
        if (writer.isOpened()) writer.release();
        if (!raw_writer_.isOpened() && !depth_writer_.isOpened()) {
            is_recording_ = false; 
            start_record_time_ = rclcpp::Time(0, 0, this->get_clock()->get_clock_type());
        }
    }
}

} // namespace booster_vision
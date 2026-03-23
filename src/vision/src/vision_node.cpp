// =================================================================================================
// Note: Please ensure the following variables are declared in your include/booster_vision/vision_node.h:
// bool is_recording_ = true;
// bool writers_initialized_ = false;
// cv::VideoWriter raw_writer_;
// cv::VideoWriter edge_writer_;
// cv::VideoWriter depth_writer_;
// rclcpp::Time start_record_time_;
// rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_img_pub_;
// =================================================================================================

#include "booster_vision/vision_node.h"

#include <functional>
#include <filesystem>
#include <iostream>
#include <algorithm>

#include <opencv2/imgproc.hpp> 
#include <yaml-cpp/yaml.h>
#include <cv_bridge/cv_bridge.h>

// [LIB] Internal Modules: Kept strictly to synchronization and image translation
#include "booster_vision/base/data_syncer.hpp"     // Matches Color + Depth images by time
#include "booster_vision/base/misc_utils.hpp"
#include "booster_vision/img_bridge.h"             // Converts ROS images to OpenCV
#include "booster_vision/model/detector.h"
#include "booster_vision/pose_estimator/pose_estimator.h"

// [LIB] Custom ROS 2 Messages for the Brain Node
#include "vision_interface/msg/detected_object.hpp"
#include "vision_interface/msg/detections.hpp"

namespace booster_vision {
// =================================================================================================
// [CONSTRUCTOR]
// Role: Creates the ROS 2 node with the given name.
// =================================================================================================
VisionNode::VisionNode(const std::string &node_name, const rclcpp::NodeOptions &options) 
    : Node(node_name, options) {
// We leave this empty because all the heavy lifting is done in the Init() function!
}

// =================================================================================================
// [FUNCTION] Init
// Role: The Setup Phase. Loads YAML configs and initializes DataSyncer without AI baggage.
// =================================================================================================
void VisionNode::Init(const std::string &cfg_template_path, const std::string &cfg_path) {
    
    // --- 1. Load Configuration ---
    if (!std::filesystem::exists(cfg_template_path)) {
        std::cerr << "Error: Configuration template file '" << cfg_path << "' does not exist." << std::endl;
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

    std::cout << "loaded file: " << std::endl << node << std::endl;

    // --- 2. Load Camera Mathematics (Calibration) ---
    if (!node["camera"]) {
        std::cerr << "no camera param found here" << std::endl;
        return;
    } else {
        camera_type_ = node["camera"]["type"].as<std::string>();
        intr_ = Intrinsics(node["camera"]["intrin"]);
        p_eye2head_ = as_or<Pose>(node["camera"]["extrin"], Pose());

        float pitch_comp = as_or<float>(node["camera"]["pitch_compensation"], 0.0);
        float yaw_comp = as_or<float>(node["camera"]["yaw_compensation"], 0.0);
        p_headprime2head_ = Pose(0, 0, 0, 0, pitch_comp * M_PI / 180, yaw_comp * M_PI / 180);
    }

    // --- 3. Initialize YOLO Detector ---
    if (!node["detection_model"]) {
        std::cerr << "Warning: No detection model param found." << std::endl;
    } else {
        // Read the backend type from the new YAML location
        std::string backend = as_or<std::string>(node["detection_model"]["backend"], "tensorrt");
        std::string active_model_path;

        if (backend == "cpu_onnx") {
            active_model_path = as_or<std::string>(node["detection_model"]["model_path_onnx"], "");
            std::cout << "Configuring for ONNX Backend using: " << active_model_path << std::endl;
            // NOTE: You will need to implement an OnnxDetector class that 
            // satisfies the detector_ interface to use the CPU backend.
        } else {
            active_model_path = as_or<std::string>(node["detection_model"]["model_path_tensor"], "");
            std::cout << "Configuring for TensorRT Backend using: " << active_model_path << std::endl;
        }

        // Initialize the detector with the chosen path
        detector_ = YoloV8Detector::CreateYoloV8Detector(node["detection_model"], active_model_path);
        
        classnames_ = node["detection_model"]["classnames"].as<std::vector<std::string>>();
        
        // Post processing filters
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

    // --- 4. Initialize Pose Estimators (2D -> 3D Projection) ---
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
    }

    // --- 5. Setup Synchronization (Replaces message_filters) ---
    use_depth_ = as_or<bool>(node["camera"]["use_depth"], false);
    is_recording_ = as_or<bool>(node["camera"]["save_data"], false);
    data_syncer_ = std::make_shared<DataSyncer>(use_depth_);

    // --- 6. ROS 2 Communication Setup ---
    std::cout << "current camera_type : " << camera_type_ << std::endl;
    
    // Dynamically loading camera/depth topics from vision.yaml
    std::string color_topic = as_or<std::string>(node["camera"]["camera_topic"], "/booster_camera_bridge/image_left_raw");
    std::string depth_topic = as_or<std::string>(node["camera"]["depth_topic"], "/booster_camera_bridge/StereoNetNode/stereonet_depth");

    std::cout << "Listening to Color: " << color_topic << std::endl;
    std::cout << "Listening to Depth: " << depth_topic << std::endl;

    it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    color_sub_ = it_->subscribe(color_topic, 1, std::bind(&VisionNode::ColorCallback, this, std::placeholders::_1));
    depth_sub_ = it_->subscribe(depth_topic, 1, std::bind(&VisionNode::DepthCallback, this, std::placeholders::_1));
    pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>("/head_pose", 10, std::bind(&VisionNode::PoseCallBack, this, std::placeholders::_1));

    // Publisher for rqt_image_view (Debugging)
    detection_pub_ = this->create_publisher<vision_interface::msg::Detections>("/booster_soccer/detection", rclcpp::QoS(1));
}

// =================================================================================================
// [FUNCTION] ProcessData
// Role: Runs AI inference, projects 2D pixels to 3D field coordinates, and publishes Detections.
// =================================================================================================
void VisionNode::ProcessData(SyncedDataBlock &synced_data, vision_interface::msg::Detections &detection_msg) {
    if (!detector_) return;

    cv::Mat color = synced_data.color_data.data;
    cv::Mat depth = synced_data.depth_data.data;

    // Convert depth for the pose estimator
    cv::Mat depth_float;
    if (!depth.empty() && depth.depth() == CV_16U) {
        depth.convertTo(depth_float, CV_32F, 0.001, 0); // Convert mm to meters
    } else {
        depth_float = depth;
    }

    // Kinematic chain: Head to Base -> Eye to Base
    Pose p_head2base = synced_data.pose_data.data;
    Pose p_eye2base = p_head2base * p_headprime2head_ * p_eye2head_;

    // 1. AI Inference
    auto detections = detector_->Inference(color);

    // Helper lambda to fetch the right estimator algorithm based on class type
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

    // 2. Post-processing (Confidence filtering & Single Ball Assumption)
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
        
        // Calculate coordinate using geometry (projection to z=0) and Depth Camera
        Pose pose_obj_by_color = pose_estimator->EstimateByColor(p_eye2base, detection, color);
        Pose pose_obj_by_depth = pose_estimator->EstimateByDepth(p_eye2base, detection, color, depth_float);

        // Populate ROS Message payload
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

        detection_msg.detected_objects.push_back(detection_obj);
    }

    // 4. Compute Image Corner Points Position (For field boundary checks in Brain)
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

    // 5. Publish to ROS 2 Topic
    detection_pub_->publish(detection_msg);
}

// =================================================================================================
// [FUNCTION] ColorCallback
// Role: Retrieves synchronized frames and passes them to ProcessData.
// =================================================================================================
void VisionNode::ColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    if (!msg) {
        std::cerr << "empty image message." << std::endl;
        return;
    }

    // 1. Convert ROS Image Message -> OpenCV Matrix
    cv::Mat img;
    try {
        img = toCVMat(*msg);
    } catch (std::exception &e) {
        std::cerr << "converting msg to cv::Mat failed: " << e.what() << std::endl;
        return;
    }

    // Ensure the image is in BGR format for OpenCV processing
    if (msg->encoding == "rgb8") {
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    }

    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    // 2. TIME MACHINE (Synchronization via DataSyncer)
    SyncedDataBlock synced_data = data_syncer_->getSyncedDataBlock(ColorDataBlock(img, timestamp));

        // Ensure we have a valid synced frame before processing
    if (synced_data.color_data.data.empty()) return;

    vision_interface::msg::Detections detection_msg;
    detection_msg.header = msg->header;
    
    ProcessData(synced_data, detection_msg);

    // ==========================================================
    // --- [K1DIY FEATURE] 5-SECOND DUAL VIDEO RECORDER ---
    // ==========================================================
    if (is_recording_) {
        cv::Mat color = synced_data.color_data.data;
        cv::Mat depth = synced_data.depth_data.data;

        // Safety check: if DataSyncer didn't find a match yet, skip processing
        if (color.empty() || depth.empty()) return;

        // Basic Image Processing (Edge Detection)
        // cv::Mat gray, edges, edges_bgr;
        // cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
        // cv::Canny(gray, edges, 50, 150);
        // cv::cvtColor(edges, edges_bgr, cv::COLOR_GRAY2BGR);

        cv::Mat depth_color;
        if (!depth.empty()) {
            // Convert 16-bit/32-bit depth to an 8-bit color map for video
            cv::Mat depth_8u;
            cv::normalize(depth, depth_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::applyColorMap(depth_8u, depth_color, cv::COLORMAP_JET);
        }
        
        std::string save_path = "data/test/";
        if (writers_initialized_) {
            int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            std::filesystem::create_directories(save_path);
            
            raw_writer_.open(save_path + "raw_video.avi", codec, 30.0, color.size(), true);
            depth_writer_.open(save_path + "depth_video.avi", codec, 30.0, depth_color.size(), true); 
            // edge_writer_.open(save_path + "edge_video.avi", codec, 30.0, edges_bgr.size(), true); 

            start_record_time_ = this->get_clock()->now();
            std::cout << ">>> RECORDING 5 SECONDS OF VIDEO TO " << save_path << " <<<" << std::endl;
            writers_initialized_ = false;
        }


        auto elapsed = this->get_clock()->now() - start_record_time_;
        if (elapsed.seconds() <= 5.0) {
            raw_writer_.write(color);
            depth_writer_.write(depth_color);
            // edge_writer_.write(edges_bgr);
            if (!depth_color.empty() && depth_writer_.isOpened()) {
                depth_writer_.write(depth_color);
            }
        } else {
            std::cout << ">>> FINISHED RECORDING! Videos saved to " << save_path << " <<<" << std::endl;
            raw_writer_.release();
            depth_writer_.release();
            // edge_writer_.release();
            if (depth_writer_.isOpened()) {
                depth_writer_.release();
            }
            is_recording_ = false;
        }
    }

    // Optional: Visual Debugging over ROS
    // cv::Mat debug_img = YoloV8Detector::DrawDetection(img, ...); 
    // Publish to debug_img_pub_ here if needed.
}

// =================================================================================================
// [FUNCTION] DepthCallback
// Role: Called when the depth camera sends data. Stores it in DataSyncer.
// =================================================================================================
void VisionNode::DepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    cv::Mat img;
    try {
        img = toCVMat(*msg);
    } catch (std::exception &e) {
        std::cerr << "cv_bridge exception " << e.what() << std::endl;
        return;
    }

    if (img.empty() || img.depth() != CV_16U) return;

    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    data_syncer_->AddDepth(DepthDataBlock(img, timestamp));
}

// =================================================================================================
// [FUNCTION] PoseCallBack
// Role: Called when the robot's motors report the head position. Stores it in DataSyncer.
// =================================================================================================
void VisionNode::PoseCallBack(const geometry_msgs::msg::Pose::SharedPtr msg) {
    auto current_time = this->get_clock()->now();
    double timestamp = static_cast<double>(current_time.nanoseconds()) * 1e-9;

    auto pose = Pose(msg->position.x, msg->position.y, msg->position.z, 
                     msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
    data_syncer_->AddPose(PoseDataBlock(pose, timestamp));
}

} // namespace booster_vision
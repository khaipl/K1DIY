#include "booster_vision/vision_node.h"

#include <functional>
#include <filesystem>
#include <iostream>
#include <algorithm>

#include <opencv2/imgproc.hpp> 
#include <opencv2/highgui.hpp> // Required for cv::imshow
#include <yaml-cpp/yaml.h>
#include <cv_bridge/cv_bridge.h>

// [LIB] Internal Modules: Kept strictly to synchronization and image translation
#include "booster_vision/base/data_syncer.hpp"     // Matches Color + Depth images by time
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
// We leave this empty because all the heavy lifting is done in the Init() function!
}

// =================================================================================================
// [FUNCTION] Init
// Role: The Setup Phase. Loads YAML configs and initializes DataSyncer without AI baggage.
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
            active_model_path = as_or<std::string>(node["detection_model"]["model_path_tensorrt"], "");
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

    // --- 4. Initialize YOLO Segmentor (Field Lines) ---
    if (!node["segmentation_model"]) {
        std::cerr << "Warning: No segmentation model param found." << std::endl;
    } else {
        std::string backend = as_or<std::string>(node["segmentation_model"]["backend"], "tensorrt");
        std::string active_seg_model_path;

        if (backend == "cpu_onnx") {
            active_seg_model_path = as_or<std::string>(node["segmentation_model"]["model_path_onnx"], "");
        } else {
            active_seg_model_path = as_or<std::string>(node["segmentation_model"]["model_path_tensorrt"], "");
        }

        segmentor_ = YoloV8Segmentor::CreateYoloV8Segmentor(node["segmentation_model"], active_seg_model_path);
        seg_data_syncer_ = std::make_shared<DataSyncer>(false); // 2D only, no depth needed for lines
    }

    // --- 5. Initialize Pose Estimators (2D -> 3D Projection) ---
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

    // --- 6. Setup Synchronization ---
    data_syncer_ = std::make_shared<DataSyncer>(true);

    /// --- 7. Load custom parameters ---
    show_det_ = as_or<bool>(node["show_det"], false);
    show_seg_ = as_or<bool>(node["show_seg"], false);
    is_recording_ = as_or<bool>(node["is_recording"], false);

    // --- 8. ROS 2 Communication Setup ---
    std::string color_topic = as_or<std::string>(node["camera"]["camera_topic"], "/booster_camera_bridge/image_left_raw");
    std::string depth_topic = as_or<std::string>(node["camera"]["depth_topic"], "/booster_camera_bridge/StereoNetNode/stereonet_depth");

    std::cout << "Listening to Color: " << color_topic << std::endl;
    std::cout << "Listening to Depth: " << depth_topic << std::endl;

    it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    
    // Detection topics
    color_sub_ = it_->subscribe(color_topic, 1, std::bind(&VisionNode::ColorCallback, this, std::placeholders::_1));
    depth_sub_ = it_->subscribe(depth_topic, 1, std::bind(&VisionNode::DepthCallback, this, std::placeholders::_1));
    pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>("/head_pose", 10, std::bind(&VisionNode::PoseCallBack, this, std::placeholders::_1));

    // Publisher for rqt_image_view (Debugging)
    detection_pub_ = this->create_publisher<vision_interface::msg::Detections>("/booster_soccer/detection", rclcpp::QoS(1));
    if (show_det_) {
        detection_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/booster_soccer/debug_det_img", rclcpp::QoS(1));
    }

    // Segmentation topics
    if (segmentor_) {
        color_seg_sub_ = it_->subscribe(color_topic, 1, std::bind(&VisionNode::SegmentationCallback, this, std::placeholders::_1));
        field_line_pub_ = this->create_publisher<vision_interface::msg::LineSegments>("/booster_soccer/line_segments", rclcpp::QoS(1));
        if (show_seg_) {
            segmentation_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/booster_soccer/debug_seg_img", rclcpp::QoS(1));
        }
    }
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
        depth.convertTo(depth_float, CV_32F, 0.001, 0); 
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

    // 4. Compute Image Corner Points Position
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
    cv::Mat det_img_out;
    if (show_det_ || is_recording_) {
        det_img_out = YoloV8Detector::DrawDetection(color, filtered_detections);
    }

    if (show_det_) {
        std_msgs::msg::Header header;
        header.stamp = this->get_clock()->now();
        sensor_msgs::msg::Image::SharedPtr debug_msg = cv_bridge::CvImage(header, "bgr8", det_img_out).toImageMsg();
        if (detection_img_pub_) {
            detection_img_pub_->publish(*debug_msg);
        }
    }

    // Save 10s video
    if (is_recording_) {
        RecordDebugVideo(raw_writer_, det_img_out, "ai_detection_video.avi", "DETECTIONS",10.0);
    } else if (raw_writer_.isOpened()) {
        raw_writer_.release();
    }

    // 5. Publish to ROS 2 Topic
    detection_pub_->publish(detection_msg);
}

// =================================================================================================
// [FUNCTION] ProcessSegmentationData
// Role: Runs Seg inference and fits field line segments for the locator (Brain node).
// =================================================================================================
void VisionNode::ProcessSegmentationData(SyncedDataBlock &synced_data, vision_interface::msg::LineSegments &field_line_segs_msg) {
    if (!segmentor_) return;

    cv::Mat color = synced_data.color_data.data;
    Pose p_head2base = synced_data.pose_data.data;
    Pose p_eye2base = p_head2base * p_headprime2head_ * p_eye2head_;

    // 1. AI Inference
    auto segmentations = segmentor_->Inference(color);
    std::vector<FieldLineSegment> field_line_segs;

    // 2. Fit Contours to 3D Field Lines
    for (auto &seg : segmentations) {
        int area_threshold = 75; // Minimum contour size
        auto line_segs = FitFieldLineSegments(p_eye2base, intr_, seg.contour, area_threshold);
        
        for (auto line_seg : line_segs) {
            float inlier_precentage = static_cast<float>(line_seg.inlier_count) / line_seg.contour_2d_points.size();
            if (inlier_precentage < 0.25) continue;

            // 3D Coordinates (For the Brain/Locator)
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[0].x);
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[0].y);
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[1].x);
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[1].y);

            // 2D UV Coordinates (For debugging)
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
    cv::Mat seg_img_out;
    if (show_seg_ || is_recording_) {
        seg_img_out = YoloV8Segmentor::DrawSegmentation(color, segmentations);
        seg_img_out = DrawFieldLineSegments(seg_img_out, field_line_segs);
        
        // Convert BACK to BGR
        cv::cvtColor(seg_img_out, seg_img_out, cv::COLOR_RGB2BGR);
    }

    if (show_seg_) {
        std_msgs::msg::Header header;
        header.stamp = this->get_clock()->now();
        sensor_msgs::msg::Image::SharedPtr debug_msg = cv_bridge::CvImage(header, "bgr8", seg_img_out).toImageMsg();
        if (segmentation_img_pub_) {
            segmentation_img_pub_->publish(*debug_msg);
        }
    }

    // Cleaned up video call!
    if (is_recording_) {
        RecordDebugVideo(depth_writer_, seg_img_out, "ai_segmentation_video.avi", "SEGMENTATION",6.0);
    } else if (depth_writer_.isOpened()) {
        depth_writer_.release();
    }

    // 3. Publish
    field_line_pub_->publish(field_line_segs_msg);
}

// =================================================================================================
// [FUNCTION] ColorCallback
// Role: Retrieves synchronized frames and passes them to ProcessData.
// =================================================================================================
void VisionNode::ColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    if (!msg) return;

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
    
    // Video writing now happens *inside* ProcessData so we get the AI output!
    ProcessData(synced_data, detection_msg);
}

// =================================================================================================
// [FUNCTION] SegmentationCallback
// =================================================================================================
void VisionNode::SegmentationCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    if (!msg || !segmentor_) return;

    cv::Mat img;
    try {
        img = toCVMat(*msg).clone();
    } catch (std::exception &e) {
        std::cerr << "cv_bridge exception: " << e.what() << std::endl;
        return;
    }

    if (msg->encoding == "rgb8") {
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    }

    vision_interface::msg::LineSegments field_line_segs_msg;
    field_line_segs_msg.header = msg->header;
    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    SyncedDataBlock synced_data = seg_data_syncer_->getSyncedDataBlock(ColorDataBlock(img, timestamp));
    
    if (synced_data.color_data.data.empty()) return;
    
    // Video writing now happens *inside* ProcessSegmentationData so we get the AI output!
    ProcessSegmentationData(synced_data, field_line_segs_msg);
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
    // Note: seg_data_syncer_ intentionally does not receive depth since lines are 2D projected.
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
    
    // Crucial: The Segmentor also needs the robot's pose to project lines to the 3D ground plane
    if (seg_data_syncer_) {
        seg_data_syncer_->AddPose(PoseDataBlock(pose, timestamp));
    }
}

// =================================================================================================
// [FUNCTION] RecordDebugVideo
// Role: Handles directory creation, codec setup, and the 5-second shared timer for video logging.
// =================================================================================================
void VisionNode::RecordDebugVideo(cv::VideoWriter& writer, const cv::Mat& frame, const std::string& filename, const std::string& log_name, double fps) {
    if (frame.empty()) return;

    std::string save_path = "data/test/";

    // 1. Initialize the writer
    if (!writer.isOpened()) {
        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        std::filesystem::create_directories(save_path);
        
        // Use the custom FPS passed into the function!
        writer.open(save_path + filename, codec, fps, frame.size(), true);
        
        if (start_record_time_.nanoseconds() == 0) {
            start_record_time_ = this->get_clock()->now();
            std::cout << ">>> RECORDING 10 SECONDS OF AI DATA TO " << save_path << " <<<" << std::endl;
        }
    }

    // 2. Check elapsed time
    auto elapsed = this->get_clock()->now() - start_record_time_;
    
    // 3. Write or Cleanup
    if (elapsed.seconds() <= 10.0) {
        writer.write(frame);
    } else {
        if (writer.isOpened()) {
            std::cout << ">>> FINISHED RECORDING " << log_name << "! <<<" << std::endl;
            writer.release();
        }
        
        // K1DIY FIX: Only shut off the master switch if BOTH writers are successfully closed
        if (!raw_writer_.isOpened() && !depth_writer_.isOpened()) {
            is_recording_ = false; 
            start_record_time_ = rclcpp::Time(0, 0, this->get_clock()->get_clock_type());
        }
    }
}

} // namespace booster_vision
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

#include <opencv2/imgproc.hpp> 
#include <yaml-cpp/yaml.h>
#include <cv_bridge/cv_bridge.h>

// [LIB] Internal Modules: Kept strictly to synchronization and image translation
#include "booster_vision/base/data_syncer.hpp"     // Matches Color + Depth images by time
#include "booster_vision/base/misc_utils.hpp"
#include "booster_vision/img_bridge.h"             // Converts ROS images to OpenCV

namespace booster_vision {
// =================================================================================================
// [CONSTRUCTOR]
// Role: Creates the ROS 2 node with the given name.
// =================================================================================================
VisionNode::VisionNode(const std::string &node_name) : Node(node_name) {
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

    // --- 3. Setup Synchronization (Replaces message_filters) ---
    use_depth_ = as_or<bool>(node["use_depth"], false);
    data_syncer_ = std::make_shared<DataSyncer>(use_depth_);

    // --- 4. ROS 2 Communication Setup ---
    std::cout << "current camera_type : " << camera_type_ << std::endl;
    
    // Dynamically loading camera/depth topics from vision.yaml
    std::string color_topic = as_or<std::string>(node["camera_topic"], "/booster_camera_bridge/image_left_raw");
    std::string depth_topic = as_or<std::string>(node["depth_topic"], "/booster_camera_bridge/StereoNetNode/stereonet_depth");

    std::cout << "Listening to Color: " << color_topic << std::endl;
    std::cout << "Listening to Depth: " << depth_topic << std::endl;

    it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    color_sub_ = it_->subscribe(color_topic, 1, std::bind(&VisionNode::ColorCallback, this, std::placeholders::_1));
    depth_sub_ = it_->subscribe(depth_topic, 1, std::bind(&VisionNode::DepthCallback, this, std::placeholders::_1));
    pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>("/head_pose", 10, std::bind(&VisionNode::PoseCallBack, this, std::placeholders::_1));

    // Publisher for rqt_image_view (Debugging)
    debug_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/vision/debug_edges", 10);
}

// =================================================================================================
// [FUNCTION] ColorCallback
// Role: Retrieves synchronized frames, processes edges, and records video.
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

    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    // 2. TIME MACHINE (Synchronization via DataSyncer)
    SyncedDataBlock synced_data = data_syncer_->getSyncedDataBlock(ColorDataBlock(img, timestamp));
    cv::Mat color = synced_data.color_data.data;
    cv::Mat depth = synced_data.depth_data.data;

    // Safety check: if DataSyncer didn't find a match yet, skip processing
    if (color.empty()) return;

    // 3. Basic Image Processing (Edge Detection)
    cv::Mat gray, edges;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 50, 150);

    // ==========================================================
    // --- [K1DIY FEATURE] 5-SECOND DUAL VIDEO RECORDER ---
    // ==========================================================
    if (is_recording_ && !depth.empty()) {
        cv::Mat depth_8u, depth_color;
        cv::Mat edges_bgr;
        
        cv::cvtColor(edges, edges_bgr, cv::COLOR_GRAY2BGR);
        cv::normalize(depth, depth_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(depth_8u, depth_color, cv::COLORMAP_JET);

        if (!writers_initialized_) {
            int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            std::string save_path = "data/test/";
            std::filesystem::create_directories(save_path);
            
            raw_writer_.open(save_path + "raw_video.avi", codec, 30.0, color.size(), true);
            edge_writer_.open(save_path + "edge_video.avi", codec, 30.0, edges_bgr.size(), true); 
            depth_writer_.open(save_path + "depth_video.avi", codec, 30.0, depth_color.size(), true); 
            
            start_record_time_ = this->get_clock()->now();
            writers_initialized_ = true;
            std::cout << ">>> RECORDING 5 SECONDS OF VIDEO TO " << save_path << " <<<" << std::endl;
        }

        auto elapsed = this->get_clock()->now() - start_record_time_;
        if (elapsed.seconds() <= 5.0) {
            raw_writer_.write(color);
            edge_writer_.write(edges_bgr);
            depth_writer_.write(depth_color);
        } else {
            std::cout << ">>> FINISHED RECORDING! Videos saved to K1DIY/data/test/ <<<" << std::endl;
            raw_writer_.release();
            edge_writer_.release();
            depth_writer_.release();
            is_recording_ = false;
        }
    }

    // 4. Visualization Publisher (For rqt_image_view)
    std_msgs::msg::Header header;
    header.stamp = msg->header.stamp; // Keep the original timestamp
    header.frame_id = "camera_link";

    try {
        auto debug_msg = cv_bridge::CvImage(header, "mono8", edges).toImageMsg();
        debug_img_pub_->publish(*debug_msg);
    } catch (cv_bridge::Exception& e) {
        std::cerr << "cv_bridge exception: " << e.what() << std::endl;
    }
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
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h> // Still needed for publishing the debug mono8 image
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <memory>
#include <vector>
#include <filesystem>

// === [NEW INCLUDES FOR SYNCHRONIZATION] ===
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

// === [NAOVAK1 HARDWARE SPECIFIC] ===
// Using the official Booster image bridge to handle Jetson NV12 camera formats
#include "booster_vision/img_bridge.h"

// Simple internal struct to hold detections until we need custom ROS messages
struct Detection {
    int class_id;
    std::string class_name;
    float confidence;
    cv::Rect bbox;
};

// ===========================================================================
// 1. STRATEGY PATTERN: THE BASE INTERFACE
// ===========================================================================
class YoloDetector {
public:
    virtual ~YoloDetector() = default;
    virtual std::vector<Detection> Inference(const cv::Mat& frame) = 0;
};

// ===========================================================================
// 2. STRATEGY A: ONNX (LAPTOP / CPU)
// ===========================================================================
class OnnxDetector : public YoloDetector {
public:
    OnnxDetector(const std::string& model_path) {
        if (model_path.empty() || !std::filesystem::exists(model_path)) {
            std::cerr << "[OnnxDetector] ERROR: Model file not found at " << model_path << ". AI disabled." << std::endl;
            return;
        }
        net_ = cv::dnn::readNetFromONNX(model_path);
    }

    std::vector<Detection> Inference(const cv::Mat& frame) override {
        std::vector<Detection> results;
        if (net_.empty() || frame.empty()) return results;

        // YOLOv8 preprocessing
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        net_.setInput(blob);

        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        // Note: Actual YOLOv8 post-processing (NMS, bounding box math) goes here!
        std::cout << "ONNX Inference executed successfully!" << std::endl;
        
        return results;
    }
private:
    cv::dnn::Net net_;
};

// ===========================================================================
// 3. STRATEGY B: TENSORRT (ROBOT / GPU)
// ===========================================================================
#ifdef USE_TENSORRT
class TrtDetector : public YoloDetector {
public:
    TrtDetector(const std::string& model_path) {
        std::cout << "[TrtDetector] Initializing TensorRT Engine from: " << model_path << std::endl;
        // TensorRT loading logic goes here
    }
    std::vector<Detection> Inference(const cv::Mat& frame) override {
        std::vector<Detection> results;
        std::cout << "TensorRT Inference executed!" << std::endl;
        return results;
    }
};
#endif

// ===========================================================================
// 4. THE MAIN ROS NODE
// ===========================================================================
class VisionNode : public rclcpp::Node {
public:
    VisionNode() : Node("vision_node") {
        
        // --- A. Load ROS 2 Parameters ---
        // New toggle for safe testing vs full AI
        this->declare_parameter<bool>("enable_ai", false);
        this->declare_parameter<std::string>("backend", "cpu_onnx");
        this->declare_parameter<std::string>("model_path", "src/vision/model/yolov8_k1.onnx");
        // Declare color topic of left eye view
        this->declare_parameter<std::string>("camera_topic", "/booster_camera_bridge/image_left_raw");
        // Declare depth topic for 3D distance
        this->declare_parameter<std::string>("depth_topic", "/booster_camera_bridge/StereoNetNode/stereonet_depth");

        bool enable_ai = this->get_parameter("enable_ai").as_bool();
        std::string backend = this->get_parameter("backend").as_string();
        std::string model_path = this->get_parameter("model_path").as_string();
        std::string camera_topic = this->get_parameter("camera_topic").as_string();
        std::string depth_topic = this->get_parameter("depth_topic").as_string();

        RCLCPP_INFO(this->get_logger(), "Starting Vision Node.");

        // --- B. AI Feature Toggle Logic ---
        if (enable_ai) {
            RCLCPP_INFO(this->get_logger(), "AI Mode ENABLED. Loading %s model...", backend.c_str());

            if (backend == "cpu_onnx") {
                detector_ = std::make_shared<OnnxDetector>(model_path);
            } 
            else if (backend == "gpu_trt") {
#ifdef USE_TENSORRT
                detector_ = std::make_shared<TrtDetector>(model_path);
#else
                RCLCPP_ERROR(this->get_logger(), "Requested GPU, but built without TensorRT! Shutting down.");
                rclcpp::shutdown();
                return;
#endif
            }
        } else {
            // SAFE MODE: Bypass AI completely for camera/edge-detection testing
            RCLCPP_INFO(this->get_logger(), "AI Mode DISABLED. Running in camera/edge-detection only mode.");
            detector_ = nullptr;
        }
        
        // Listen to the robot's camera (UPDATED FOR STEREO SYNC)
        auto qos = rclcpp::QoS(rclcpp::SensorDataQoS());
        color_sub_.subscribe(this, camera_topic, qos.get_rmw_qos_profile());
        depth_sub_.subscribe(this, depth_topic, qos.get_rmw_qos_profile());

        // ApproximateTime policy to pair Color and Depth frames together even if hardware is slightly out of sync
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), color_sub_, depth_sub_);
        sync_->registerCallback(std::bind(&VisionNode::ColorDepthCallback, this, std::placeholders::_1, std::placeholders::_2));

        // Publisher for the laptop to view the edges
        debug_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/vision/debug_edges", 10);
    }

private:
    void ColorDepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr& color_msg,
                            const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) {
        
        // 1. Convert ROS Image -> OpenCV Mat using the K1's specific image bridge
        cv::Mat frame, depth_frame;
        try {
            // This safely handles the NV12 hardware encoding from the Jetson
            frame = booster_vision::toCVMat(*color_msg);
            // Handles MONO16 endianness for depth
            depth_frame = booster_vision::toCVMat(*depth_msg); 
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "booster_vision::img_bridge exception: %s", e.what());
            return;
        }

        if (frame.empty() || depth_frame.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty frame from img_bridge.");
            return;
        }

        // 2. Basic Image Processing (Edge Detection)
        cv::Mat gray, edges;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edges, 50, 150);

        // --- NEW: Process Depth Color Map (Distance to Color) ---
        cv::Mat depth_8u, depth_color;
        // Squashes the 16-bit millimeter data down to an 8-bit scale
        cv::normalize(depth_frame, depth_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
        // Applies the Heatmap (Red = close, Blue = far)
        cv::applyColorMap(depth_8u, depth_color, cv::COLORMAP_JET);


        // ==========================================================
        // --- NEW FEATURE: 5-SECOND DUAL VIDEO RECORDER ---
        // (Now upgraded to Triple video recorder for Depth!)
        // ==========================================================
        if (is_recording_) {
            // Initialize writers on the very first frame to get the correct resolution
            if (!writers_initialized_) {
                int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                
                // --- NEW: Ensure the DIY data directory exists ---
                std::string save_path = "data/test/";
                std::filesystem::create_directories(save_path);
                
                // Saving to the new path in K1DIY/data/test/
                raw_writer_.open(save_path + "raw_video.avi", codec, 30.0, frame.size(), true);
                edge_writer_.open(save_path + "edge_video.avi", codec, 30.0, frame.size(), true); 
                depth_writer_.open(save_path + "depth_video.avi", codec, 30.0, depth_color.size(), true); 
                
                start_record_time_ = this->get_clock()->now();
                writers_initialized_ = true;
                RCLCPP_INFO(this->get_logger(), ">>> RECORDING 5 SECONDS OF VIDEO TO %s <<<", save_path.c_str());
            }

            // Calculate how much time has passed
            auto elapsed = this->get_clock()->now() - start_record_time_;

            if (elapsed.seconds() <= 5.0) {
                // Write the raw color frame
                raw_writer_.write(frame);
                
                // Convert 1-channel Edge image to 3-channel BGR so the VideoWriter doesn't crash
                cv::Mat edges_bgr;
                cv::cvtColor(edges, edges_bgr, cv::COLOR_GRAY2BGR);
                edge_writer_.write(edges_bgr);

                // Write the colorized depth map
                depth_writer_.write(depth_color);
            } else {
                // Stop recording after 5 seconds!
                RCLCPP_INFO(this->get_logger(), ">>> FINISHED RECORDING! Videos saved to K1DIY/data/test/ <<<");
                raw_writer_.release();
                edge_writer_.release();
                depth_writer_.release();
                is_recording_ = false;
            }
        }
        // ==========================================================

        if (detector_) {
            try { auto detections = detector_->Inference(frame); }
            catch (const std::exception& e) { RCLCPP_WARN_ONCE(this->get_logger(), "AI Inference failed"); }
        }

        // 4. Visualization (ROBOT HEADLESS MODE)
        std_msgs::msg::Header header;
        header.stamp = this->get_clock()->now();
        header.frame_id = "camera_link";

        try {
            // We can still use cv_bridge here because we are encoding a simple mono8 image to send OUT
            auto debug_msg = cv_bridge::CvImage(header, "mono8", edges).toImageMsg();
            debug_img_pub_->publish(*debug_msg);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    std::shared_ptr<YoloDetector> detector_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_img_pub_;

    // --- Sync Policy Definitions for Dual-Camera ---
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
    message_filters::Subscriber<sensor_msgs::msg::Image> color_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // Variables for the video recorder
    cv::VideoWriter raw_writer_;
    cv::VideoWriter edge_writer_;
    cv::VideoWriter depth_writer_;
    rclcpp::Time start_record_time_;
    bool is_recording_ = true;
    bool writers_initialized_ = false;
};

// ===========================================================================
// 5. MAIN ENTRY POINT
// ===========================================================================
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
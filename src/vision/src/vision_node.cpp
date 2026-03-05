#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h> // Still needed for publishing the debug mono8 image
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <memory>
#include <vector>
#include <filesystem>

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
        // The default topic on the K1 might be different, but we'll read it from vision.yaml
        this->declare_parameter<std::string>("camera_topic", "/camera/image_raw");

        bool enable_ai = this->get_parameter("enable_ai").as_bool();
        std::string backend = this->get_parameter("backend").as_string();
        std::string model_path = this->get_parameter("model_path").as_string();
        std::string camera_topic = this->get_parameter("camera_topic").as_string();

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
        
        // Listen to the robot's camera
        color_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            camera_topic, 10, 
            std::bind(&VisionNode::ColorCallback, this, std::placeholders::_1)
        );

        // Publisher for the laptop to view the edges
        debug_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/vision/debug_edges", 10);
    }

private:
    void ColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        // 1. Convert ROS Image -> OpenCV Mat using the K1's specific image bridge
        cv::Mat frame;
        try {
            // This safely handles the NV12 hardware encoding from the Jetson
            frame = booster_vision::toCVMat(*msg); 
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "booster_vision::img_bridge exception: %s", e.what());
            return;
        }

        if (frame.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty frame from img_bridge.");
            return;
        }

        // 2. Basic Image Processing (Edge Detection)
        cv::Mat gray, edges;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edges, 50, 150); 

        // 3. AI Inference (Protected by Toggle)
        // If enable_ai is false, detector_ is null, and this block is safely skipped.
        if (detector_) {
            try {
                auto detections = detector_->Inference(frame);
            } catch (const std::exception& e) {
                RCLCPP_WARN_ONCE(this->get_logger(), "AI Inference failed, skipping...");
            }
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
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception during publish: %s", e.what());
        }
    }

    std::shared_ptr<YoloDetector> detector_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_img_pub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <memory>
#include <vector>
#include <filesystem>

// ===========================================================================
// [ROBOCUP DEPENDENCIES - COMMENTED OUT FOR DIY]
// These are the internal Booster modules we are currently bypassing.
// #include "vision_interface/msg/detected_object.hpp"
// #include "vision_interface/msg/detections.hpp"
// #include "booster_vision/base/data_syncer.hpp"
// #include "booster_vision/pose_estimator/pose_estimator.h"
// ===========================================================================

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

        // --- C. ROS 2 Subscriptions ---
        color_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            camera_topic, 10, 
            std::bind(&VisionNode::ColorCallback, this, std::placeholders::_1)
        );
    }

private:
    void ColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        // 1. Convert ROS Image -> OpenCV Mat
        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
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

        // 4. Visualization
        cv::imshow("NaovaK1 Laptop Feed", frame);
        cv::imshow("Field Line Detection (Edges)", edges);
        cv::waitKey(1);
    }

    std::shared_ptr<YoloDetector> detector_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
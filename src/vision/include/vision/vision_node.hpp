#ifndef VISION_NODE_HPP_
#define VISION_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

// Forward declaration of the detection struct
struct Detection {
    int class_id;
    std::string class_name;
    float confidence;
    cv::Rect bbox;
};

// ===========================================================================
// THE STRATEGY INTERFACE
// ===========================================================================
class YoloDetector {
public:
    virtual ~YoloDetector() = default;
    virtual std::vector<Detection> Inference(const cv::Mat& frame) = 0;
};

// ===========================================================================
// THE VISION NODE CLASS
// ===========================================================================
class VisionNode : public rclcpp::Node {
public:
    VisionNode();

private:
    // The main callback that processes camera images
    void ColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

    // AI Detector (can be OnnxDetector or TrtDetector at runtime)
    std::shared_ptr<YoloDetector> detector_;

    // ROS 2 Subscriber
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
};

#endif // VISION_NODE_HPP_
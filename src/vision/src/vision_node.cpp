// 1. Include our new header file!
#include "vision/vision_node.hpp"

// 2. We use BasicVisionNode:: to tell the compiler: 
// "Hey, remember that constructor we promised in the header? Here is the code for it!"
BasicVisionNode::BasicVisionNode() : Node("vision_node") {
    
    this->declare_parameter<std::string>("image_path", "");
    std::string image_path;
    this->get_parameter("image_path", image_path);

    RCLCPP_INFO(this->get_logger(), "Attempting to load image from: %s", image_path.c_str());

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load image! Check the path in your YAML file.");
        return;
    }

    cv::Mat gray, edges;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 50, 150); 

    cv::imshow("Original RoboCup Image", image);
    cv::imshow("Canny Edge Lines", edges);
    cv::waitKey(0); 
}

// 3. The main function stays exactly the same
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BasicVisionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
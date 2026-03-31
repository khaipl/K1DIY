#include <memory>
#include <string>
#include <cmath>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "vision_interface/msg/detections.hpp"

using std::placeholders::_1;

class MinimalBrain : public rclcpp::Node
{
public:
    MinimalBrain() : Node("brain_node")
    {
        // Declare parameter for multi-robot namespacing
        this->declare_parameter<std::string>("robot_name", "");
        std::string robot_name = this->get_parameter("robot_name").as_string();
        std::string topic_suffix = robot_name.empty() ? "" : "/" + robot_name;

        // Set up the vision subscriber
        std::string vision_topic = "/booster_soccer/detection" + topic_suffix;
        RCLCPP_INFO(this->get_logger(), "Starting K1DIY Brain Node...");
        RCLCPP_INFO(this->get_logger(), "Subscribing to vision topic: %s", vision_topic.c_str());
        
        vision_sub_ = this->create_subscription<vision_interface::msg::Detections>(
            vision_topic, 10, std::bind(&MinimalBrain::vision_callback, this, _1));
    }

private:
    // Struct to hold our processed ball data
    struct BallData {
        bool detected = false;
        double x_to_robot = 0.0;
        double y_to_robot = 0.0;
        double range = 0.0;
        double yaw_to_robot = 0.0;
        double confidence = 0.0;
    };

    BallData current_ball_;

    void vision_callback(const vision_interface::msg::Detections::SharedPtr msg)
    {
        double best_confidence = 0.0;
        int real_ball_index = -1;

        // 1. Parse the incoming detections to find the most confident ball
        for (size_t i = 0; i < msg->detected_objects.size(); i++)
        {
            auto obj = msg->detected_objects[i];
            
            // Check label and ensure position array has at least x and y
            if (obj.label == "Ball" && obj.position_projection.size() >= 2)
            {
                // Basic sanity check: reject false positives in the sky/horizon (X < -0.5m or X > 15.0m)
                if (obj.position_projection[0] < -0.5 || obj.position_projection[0] > 15.0) {
                    continue;
                }

                // Keep the ball with the highest confidence
                if (obj.confidence > best_confidence) {
                    best_confidence = obj.confidence;
                    real_ball_index = i;
                }
            }
        }

        // 2. Process the chosen ball
        if (real_ball_index >= 0)
        {
            auto ball_obj = msg->detected_objects[real_ball_index];
            current_ball_.detected = true;
            current_ball_.confidence = ball_obj.confidence;
            
            // The position_projection array holds the 3D estimated coordinates
            current_ball_.x_to_robot = ball_obj.position_projection[0];
            current_ball_.y_to_robot = ball_obj.position_projection[1];
            
            // Calculate distance (range) and angle (yaw)
            current_ball_.range = std::hypot(current_ball_.x_to_robot, current_ball_.y_to_robot);
            current_ball_.yaw_to_robot = std::atan2(current_ball_.y_to_robot, current_ball_.x_to_robot);

            RCLCPP_INFO(this->get_logger(), 
                "BALL SEEN | Conf: %05.1f%% | X: %04.2fm | Y: %04.2fm | Range: %04.2fm | Yaw: %04.2frad", 
                current_ball_.confidence, 
                current_ball_.x_to_robot, 
                current_ball_.y_to_robot, 
                current_ball_.range, 
                current_ball_.yaw_to_robot);
        }
        else
        {
            if (current_ball_.detected) {
                RCLCPP_WARN(this->get_logger(), "BALL LOST");
            }
            current_ball_.detected = false;
        }
    }

    rclcpp::Subscription<vision_interface::msg::Detections>::SharedPtr vision_sub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalBrain>());
    rclcpp::shutdown();
    return 0;
}
#include <memory>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "vision_interface/msg/detections.hpp"

// --- Locomotion Headers ---
#include "booster_msgs/msg/rpc_req_msg.hpp"
#include "booster_interface/msg/booster_api_req_msg.hpp"
#include "message_utils.hpp"
#include "third_party/nlohmann_json/json.hpp" 

using std::placeholders::_1;
using namespace std::chrono_literals;

class MinimalBrain : public rclcpp::Node
{
public:
    MinimalBrain() : Node("brain_node"), uuid_counter_(0)
    {
        this->declare_parameter<std::string>("robot_name", "");
        std::string robot_name = this->get_parameter("robot_name").as_string();
        std::string topic_suffix = robot_name.empty() ? "" : "/" + robot_name;

        RCLCPP_INFO(this->get_logger(), "Starting K1DIY Brain Node (CHASE MODE)...");

        // 1. Setup the Vision Subscriber (The Eyes)
        std::string vision_topic = "/booster_soccer/detection" + topic_suffix;
        vision_sub_ = this->create_subscription<vision_interface::msg::Detections>(
            vision_topic, 10, std::bind(&MinimalBrain::vision_callback, this, _1));

        // 2. Setup the Locomotion Publisher (The Muscles)
        std::string loco_topic = "LocoApiTopic" + topic_suffix + "Req";
        loco_pub_ = this->create_publisher<booster_msgs::msg::RpcReqMsg>(loco_topic, 10);
        
        // 3. Setup the Control Loop Timer (4 times a second!)
        timer_ = this->create_wall_timer(
            250ms, std::bind(&MinimalBrain::control_loop, this)); 
    }

private:
    struct BallData {
        bool detected = false;
        double x_to_robot = 0.0;
        double y_to_robot = 0.0;
        double range = 0.0;
        double yaw_to_robot = 0.0;
        double confidence = 0.0;
    };

    BallData current_ball_;
    rclcpp::Subscription<vision_interface::msg::Detections>::SharedPtr vision_sub_;

    void vision_callback(const vision_interface::msg::Detections::SharedPtr msg)
    {
        double best_confidence = 0.0;
        int real_ball_index = -1;

        for (size_t i = 0; i < msg->detected_objects.size(); i++) {
            auto obj = msg->detected_objects[i];
            if (obj.label == "Ball" && obj.position_projection.size() >= 2) {
                if (obj.position_projection[0] < -0.5 || obj.position_projection[0] > 15.0) continue;
                if (obj.confidence > best_confidence) {
                    best_confidence = obj.confidence;
                    real_ball_index = i;
                }
            }
        }

        if (real_ball_index >= 0) {
            auto ball_obj = msg->detected_objects[real_ball_index];
            current_ball_.detected = true;
            current_ball_.confidence = ball_obj.confidence;
            current_ball_.x_to_robot = ball_obj.position_projection[0];
            current_ball_.y_to_robot = ball_obj.position_projection[1];
            current_ball_.range = std::hypot(current_ball_.x_to_robot, current_ball_.y_to_robot);
            current_ball_.yaw_to_robot = std::atan2(current_ball_.y_to_robot, current_ball_.x_to_robot);
        } else {
            current_ball_.detected = false;
        }
    }

    rclcpp::Publisher<booster_msgs::msg::RpcReqMsg>::SharedPtr loco_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    int uuid_counter_;

    double cap(double value, double max_val, double min_val) {
        return std::max(min_val, std::min(value, max_val));
    }

    int call(booster_interface::msg::BoosterApiReqMsg msg) {
        auto message = booster_msgs::msg::RpcReqMsg();
        message.uuid = "k1diy-chase-" + std::to_string(++uuid_counter_);
        nlohmann::json req_header;
        req_header["api_id"] = msg.api_id;
        message.header = req_header.dump();
        message.body = msg.body;
        loco_pub_->publish(message);
        return 0;
    }

    int setVelocity(double x, double y, double theta) {
        double minx = 0.3, miny = 0.3, mintheta = 0.25;
        double vx_limit = 1.0, vy_limit = 0.4, vtheta_limit = 1.2;

        if (fabs(x) < minx && fabs(x) > 1e-5) x = (x > 0) ? minx : -minx;
        if (fabs(y) < miny && fabs(y) > 1e-5) y = (y > 0) ? miny : -miny;
        if (fabs(theta) < mintheta && fabs(theta) > 1e-5) theta = (theta > 0) ? mintheta : -mintheta;

        x = cap(x, vx_limit, -vx_limit);
        y = cap(y, vy_limit, -vy_limit);
        theta = cap(theta, vtheta_limit, -vtheta_limit);

        return call(booster_interface::CreateMoveMsg(x, y, theta));
    }

    // =============================================================
    // THE NEW INTELLIGENCE: The Chase Loop
    // =============================================================
    void control_loop() {
        if (current_ball_.detected) {
            // Proportional Control Gains
            double Kp_x = 0.8;      
            double Kp_theta = 1.2;  

            double target_distance = 0.3; 

            // Calculate speeds based on the ball's coordinates
            double vx = Kp_x * (current_ball_.x_to_robot - target_distance);
            double vtheta = Kp_theta * current_ball_.yaw_to_robot;

            RCLCPP_INFO(this->get_logger(), "CHASING | X: %.2fm, Y: %.2fm -> Cmd: vx=%.2f, vtheta=%.2f", 
                        current_ball_.x_to_robot, current_ball_.y_to_robot, vx, vtheta);
            
            setVelocity(vx, 0.0, vtheta);
        } else {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "LOST BALL | Freezing motors.");
            setVelocity(0.0, 0.0, 0.0);
        }
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalBrain>());
    rclcpp::shutdown();
    return 0;
}
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

// Ensure M_PI is defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

        RCLCPP_INFO(this->get_logger(), "Starting K1DIY Brain Node (TRUE AXIS + SMOOTH HEAD)...");

        // 1. Setup the Vision Subscriber (The Eyes)
        std::string vision_topic = "/booster_soccer/detection" + topic_suffix;
        vision_sub_ = this->create_subscription<vision_interface::msg::Detections>(
            vision_topic, 10, std::bind(&MinimalBrain::vision_callback, this, _1));

        // 2. Setup the Locomotion Publisher (The Muscles)
        std::string loco_topic = "/LocoApiTopic" + topic_suffix + "Req";
        loco_pub_ = this->create_publisher<booster_msgs::msg::RpcReqMsg>(loco_topic, 10);
        
        // 3. Setup the Control Loop Timer (4 times a second)
        timer_ = this->create_wall_timer(
            50ms, std::bind(&MinimalBrain::control_loop, this)); 
    }

private:
    enum class BrainState {
        STARTUP,
        PREPARING,
        WALKING
    };
    BrainState current_state_ = BrainState::STARTUP;
    rclcpp::Time state_change_time_;

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

    // --- Head Smoothing Memory ---
    float current_head_pitch_ = 0.3f; // Start looking slightly down
    float current_head_yaw_ = 0.0f;   // Start looking straight ahead
    float head_smoothing_ = 0.005f;    // 0.0 to 1.0. Lower = smoother/slower. 

    double last_vision_x_ = -999.0;
    double last_vision_y_ = -999.0;

    void vision_callback(const vision_interface::msg::Detections::SharedPtr msg)
    {
        double best_confidence = 0.0;
        int real_ball_index = -1;

        for (size_t i = 0; i < msg->detected_objects.size(); i++) {
            auto obj = msg->detected_objects[i];
            if (obj.label == "Ball" && obj.position_projection.size() >= 2) {
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
            
            // 1. Get the Ball's Global Coordinates from Vision
            double world_ball_x = ball_obj.position_projection[0];
            double world_ball_y = ball_obj.position_projection[1];

            RCLCPP_INFO(this->get_logger(), "--------------------------------------------------");
            RCLCPP_INFO(this->get_logger(), "[DEBUG STEP 4: Brain Received] World Ball X: %.3fm, Y: %.3fm", world_ball_x, world_ball_y);

            // 2. Get the Robot's Global Coordinates (Published by VisionNode)
            double robot_x = ball_obj.received_pos[0];
            double robot_y = ball_obj.received_pos[1];

            // Note: received_pos[5] is Yaw published in DEGREES. Convert to Radians.
            double robot_yaw = ball_obj.received_pos[5] * M_PI / 180.0;

            RCLCPP_INFO(this->get_logger(), "[DEBUG STEP 5: Robot Pose] X: %.3f, Y: %.3f, Yaw: %.3frad", robot_x, robot_y, robot_yaw);

            // 3. Calculate the absolute difference in the world
            double dx = world_ball_x - robot_x;
            double dy = world_ball_y - robot_y;

            // 4. Counter-rotate the difference by the robot's Yaw to make it Relative
            current_ball_.x_to_robot = dx * std::cos(-robot_yaw) - dy * std::sin(-robot_yaw);
            current_ball_.y_to_robot = dx * std::sin(-robot_yaw) + dy * std::cos(-robot_yaw);

            current_ball_.range = std::hypot(current_ball_.x_to_robot, current_ball_.y_to_robot);
            current_ball_.yaw_to_robot = std::atan2(current_ball_.y_to_robot, current_ball_.x_to_robot);

            RCLCPP_INFO(this->get_logger(), "[DEBUG STEP 6: Final Relative] Fwd X: %.3fm, L/R Y: %.3fm, Range: %.3fm", 
                        current_ball_.x_to_robot, current_ball_.y_to_robot, current_ball_.range);
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

    int setHeadPosition(float target_pitch, float target_yaw) {
        // 1. Define hardware limits based on the K1 Joint Table (in degrees)
        float max_pitch_deg = 38.0f;
        float min_pitch_deg = -12.0f;
        
        float max_yaw_deg = 50.0f;
        float min_yaw_deg = -50.0f;

        // 2. Convert degrees to radians (API expects radians)
        float max_pitch = max_pitch_deg * M_PI / 180.0f; 
        float min_pitch = min_pitch_deg * M_PI / 180.0f; 
        
        float max_yaw = max_yaw_deg * M_PI / 180.0f;     
        float min_yaw = min_yaw_deg * M_PI / 180.0f;     

        // 3. Cap the requested angles to strictly prevent hardware damage
        float safe_pitch = cap(target_pitch, max_pitch, min_pitch);
        float safe_yaw = cap(target_yaw, max_yaw, min_yaw);

        // 4. Send the RPC request to the robot
        return call(booster_interface::CreateRotateHeadMsg(safe_pitch, safe_yaw));
    }

    int setVelocity(double x, double y, double theta) {
        double minx = 0.3, miny = 0.1, mintheta = 0.1;
        double vx_limit = 0.5, vy_limit = 0.2, vtheta_limit = 0.3;

        if (fabs(x) < minx && fabs(x) > 1e-5) x = (x > 0) ? minx : -minx;
        if (fabs(y) < miny && fabs(y) > 1e-5) y = (y > 0) ? miny : -miny;
        if (fabs(theta) < mintheta && fabs(theta) > 1e-5) theta = (theta > 0) ? mintheta : -mintheta;

        x = cap(x, vx_limit, -vx_limit);
        y = cap(y, vy_limit, -vy_limit);
        theta = cap(theta, vtheta_limit, -vtheta_limit);

        return call(booster_interface::CreateMoveMsg(x, y, theta));
    }

    // =============================================================
    // THE NEW INTELLIGENCE: The Chase Loop with SMOOTH Head Tracking
    // =============================================================
    void control_loop() {
        auto now = this->get_clock()->now();

        switch (current_state_) {
            case BrainState::STARTUP:
                RCLCPP_INFO(this->get_logger(), "Sending kPrepare mode...");
                call(booster_interface::CreateChangeModeMsg(booster::robot::RobotMode::kPrepare));
                current_state_ = BrainState::PREPARING;
                state_change_time_ = now;
                break;

            case BrainState::PREPARING:
                // Give the physical hardware 3 seconds to stand up
                if ((now - state_change_time_).seconds() > 3.0) {
                    RCLCPP_INFO(this->get_logger(), "Sending kWalking mode...");
                    call(booster_interface::CreateChangeModeMsg(booster::robot::RobotMode::kWalking));
                    current_state_ = BrainState::WALKING;

                    state_change_time_ = now;
                }
                break;

            case BrainState::WALKING: {
                float target_pitch = 0.3f;
                float target_yaw = 0.0f;

                if (current_ball_.detected) {
                    // 1. HEAD TRACKING MATH (Where we WANT to look)
                    float camera_height = 0.87f; // K1's approximate camera height in meters
                    
                    target_yaw = current_ball_.yaw_to_robot;
                    target_pitch = std::atan2(camera_height, current_ball_.range);
                    
                    // 2. LOCOMOTION CHASE LOGIC
                    double Kp_x = 0.0;      
                    double Kp_theta = 0.0;  
                    double target_distance = 0.1; 

                    double vx = Kp_x * (current_ball_.x_to_robot - target_distance);
                    double vtheta = Kp_theta * current_ball_.yaw_to_robot;
                    
                    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, 
                        "BALL SEEN | X(Fwd): %04.2fm | Y(L/R): %04.2fm | PitchCmd: %04.2frad | YawCmd: %04.2frad", 
                        current_ball_.x_to_robot, current_ball_.y_to_robot, target_pitch, target_yaw);
                    
                    setVelocity(vx, 0.0, vtheta); 
                } else {
                    // 1. SCANNING MATH (Ball lost, sweep smoothly left/right)
                    target_pitch = 0.2f; // Keep looking down at the grass
                    
                    // Use a sine wave based on time to sweep back and forth
                    double time_sec = now.nanoseconds() * 1e-9;
                    target_yaw = 0.6f * std::sin(time_sec * 3.0); 

                    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "LOST BALL | Scanning..."); 
                    
                    setVelocity(0.0, 0.0, 0.0);
                }

                // ==========================================
                // THE SMOOTHING FILTER (Low-Pass Filter)
                // ==========================================
                // current = current + alpha * (target - current)
                current_head_pitch_ += head_smoothing_ * (target_pitch - current_head_pitch_);
                current_head_yaw_ += head_smoothing_ * (target_yaw - current_head_yaw_);

                // Send the beautifully smoothed angles to the hardware
                // setHeadPosition(current_head_pitch_, current_head_yaw_);

                // --- 3-Second Linear Move from 0.0 to 0.4 ---
                double elapsed_walking = (now - state_change_time_).seconds();
                
                // Divide elapsed time by 3.0 seconds, multiply by target (0.4), and cap it at 0.4 max
                float pitch_cmd = std::min(0.75f, static_cast<float>((elapsed_walking / 6.0) * 0.75f));
                
                setHeadPosition(pitch_cmd, 0.0f);
                
                break;
            }
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
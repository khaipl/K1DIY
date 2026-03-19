#include <rclcpp/rclcpp.hpp>
#include "booster_vision/vision_node.h"

int main(int argc, char **argv) {
    // 1. Initialize ROS 2
    rclcpp::init(argc, argv);

    // 2. Create the Vision Node
    // We pass "vision_node" as the name, which the VisionNode constructor expects
    auto node = std::make_shared<booster_vision::VisionNode>("vision_node");

    // 3. Initialize the Node with your configuration files
    // The NaovaK1 Init() function requires the paths to the YAML config files.
    // We point it directly to your K1DIY vision.yaml.
    std::string config_path = "src/vision/config/vision.yaml";
    
    // We pass it twice because NaovaK1 supports a "template" and a "local override" config.
    // Since we just have one, we pass the same path for both.
    node->Init(config_path, config_path);

    // 4. Keep the node running and listening to the camera streams
    std::cout << ">>> K1DIY Perception Node is running! Waiting for camera topics... <<<" << std::endl;
    rclcpp::spin(node);

    // 5. Clean shutdown when you press Ctrl+C
    rclcpp::shutdown();
    return 0;
}
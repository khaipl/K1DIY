cmake_minimum_required(VERSION 3.10)
project(vision)

# Compile using C++17
set(CMAKE_CXX_STANDARD 17)

# Find essential ROS 2 and OpenCV packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV 4.5 REQUIRED)

# Build the executable from our single C++ file
add_executable(vision_node src/vision_node.cpp)

# Only give these headers to vision_node. Nothing else.
target_include_directories(vision_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
)

# Link OpenCV and ROS 2 to our node
target_link_libraries(vision_node ${OpenCV_LIBRARIES})

ament_target_dependencies(vision_node rclcpp)

# Install rules so ROS 2 can find the compiled node and config files
install(TARGETS vision_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

ament_package()
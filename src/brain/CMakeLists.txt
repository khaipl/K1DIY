cmake_minimum_required(VERSION 3.8)
project(brain)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(vision_interface REQUIRED)
find_package(std_msgs REQUIRED)

# Add the executable
add_executable(brain_node src/brain_node.cpp)
target_include_directories(brain_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(brain_node PUBLIC c_std_99 cxx_std_17)

# Link dependencies
ament_target_dependencies(
  brain_node
  "rclcpp"
  "vision_interface"
  "std_msgs"
)

# Install executable
install(TARGETS brain_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
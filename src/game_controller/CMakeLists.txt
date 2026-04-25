cmake_minimum_required(VERSION 3.10)
project(game_controller)

option(NO_CUDA "Build without CUDA" ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(game_controller_interface REQUIRED)

# Add the executable
add_executable(game_controller_node src/main.cpp src/game_controller_node.cpp)
target_include_directories(game_controller_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(game_controller_node PUBLIC c_std_99 cxx_std_17)

# Link dependencies
ament_target_dependencies(
  game_controller_node
  "rclcpp"
  "std_msgs"
  "game_controller_interface"
)

# Install directories
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

# Install executable
install(TARGETS game_controller_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
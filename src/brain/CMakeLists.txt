cmake_minimum_required(VERSION 3.16)
project(brain)

option(NO_CUDA "Build without CUDA" ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
endif()

# Find all required ROS 2 and system dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(booster_interface REQUIRED)
find_package(game_controller_interface REQUIRED)
find_package(vision_interface REQUIRED)
find_package(booster_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

# Generate custom messages (Kick.msg)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Kick.msg"
  DEPENDENCIES std_msgs
)
ament_export_dependencies(rosidl_default_runtime)

# SOURCE FILES: This dynamically finds brain.cpp, brain_tree.cpp, etc. (Fixes your error!)
file(GLOB SOURCE_FILES src/*.cpp)
add_executable(brain_node ${SOURCE_FILES})

# Link the typesupport target for custom messages
rosidl_get_typesupport_target(typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(brain_node ${typesupport_target})

set_property(TARGET brain_node PROPERTY CXX_STANDARD 17)

# Link external libraries
target_link_libraries(brain_node yaml-cpp ${OpenCV_LIBS}) 

include_directories(${geometry_msgs_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})
include_directories(${sensor_msgs_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Wno-unused-parameter -Wno-sign-compare -Wno-missing-field-initializers -Wno-unused-variable")

# Link all ROS 2 package dependencies
ament_target_dependencies(brain_node 
  rclcpp 
  rosidl_typesupport_cpp 
  std_msgs 
  sensor_msgs 
  nav_msgs 
  diagnostic_msgs 
  behaviortree_cpp 
  booster_interface 
  vision_interface 
  game_controller_interface 
  booster_msgs 
  tf2 
  tf2_ros 
  tf2_geometry_msgs 
  geometry_msgs 
  visualization_msgs
)

target_include_directories(brain_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Install the executable
install(TARGETS brain_node
  DESTINATION lib/${PROJECT_NAME})

# Install the necessary directories (XML trees, configs, launch files)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY behavior_trees
  DESTINATION share/${PROJECT_NAME})

ament_package()
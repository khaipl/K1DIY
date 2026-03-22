# 1. Enter the subfolder to build the library first
add_subdirectory(base)
add_subdirectory(model)
add_subdirectory(pose_estimator)

# If running on the physical robot (ARM64), compile the calibration tools
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  add_subdirectory(calibration)
endif()

# 2. Build the node (Notice: no more BASE_SOURCES here!)
add_executable(vision_node 
    main.cpp 
    vision_node.cpp 
    img_bridge.cpp
)

# 3. Link everything together
target_link_libraries(vision_node 
    booster_vision_base
    booster_vision_pose_estimator
    yolov8_model
)

# Link ROS 2 dependencies
ament_target_dependencies(vision_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  image_transport
  tf2
  tf2_geometry_msgs
  builtin_interfaces
  vision_interface
  cv_bridge
)

# Install the executable so ROS 2 can find it
install(TARGETS vision_node DESTINATION lib/${PROJECT_NAME})
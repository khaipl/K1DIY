# 1. Enter the subfolder to build the library first
add_subdirectory(base)

# 2. Build the node (Notice: no more BASE_SOURCES here!)
add_executable(vision_node 
    main.cpp 
    vision_node.cpp 
    img_bridge.cpp
)

# 3. Link everything together
target_link_libraries(vision_node 
    booster_vision_base
    ${OpenCV_LIBRARIES} 
    yaml-cpp
)

ament_target_dependencies(vision_node 
    rclcpp
    sensor_msgs
    cv_bridge
    image_transport
)

install(TARGETS vision_node DESTINATION lib/${PROJECT_NAME})
cmake_minimum_required(VERSION 3.10)
project(vision)

# --- FIX 1: Silence CUDA Policy CMP0074 Warning ---
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# Compile using C++17
set(CMAKE_CXX_STANDARD 17)

# ==========================================
# 1. CORE DEPENDENCIES (Required)
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED) 
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)

# ==========================================
# 2. OPTIONAL GPU DEPENDENCIES (The "Shield")
# ==========================================
# We use QUIET so CMake doesn't fail if you are on your laptop
find_package(CUDA QUIET)

# TensorRT is usually found by looking directly for its C++ library (nvinfer)
find_library(TENSORRT_LIBRARY nvinfer)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h)

# ==========================================
# 3. BUILD THE EXECUTABLE
# ==========================================
add_executable(vision_node 
  src/vision_node.cpp
  src/img_bridge.cpp
)

# Prioritize OpenCV headers to avoid Jetpack version conflicts
target_include_directories(vision_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
)

# Link OpenCV and ROS 2 dependencies
target_link_libraries(vision_node ${OpenCV_LIBRARIES})

# Link ROS 2 packages including message_filters
ament_target_dependencies(vision_node 
  rclcpp
  sensor_msgs
  cv_bridge
  message_filters
)

# ==========================================
# 4. APPLY THE GPU SHIELD
# ==========================================
# If CMake finds TensorRT (Jetson), it activates GPU support.
# If not (Laptop), it safely skips this block.
if(CUDA_FOUND AND TENSORRT_LIBRARY AND TENSORRT_INCLUDE_DIR)
    message(STATUS "TensorRT found! Enabling GPU acceleration support (USE_TENSORRT).")
    target_compile_definitions(vision_node PRIVATE USE_TENSORRT) 
    target_include_directories(vision_node PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(vision_node ${TENSORRT_LIBRARY})
else()
    message(STATUS "TensorRT NOT found. Building for CPU/ONNX fallback only.")
endif()

# ==========================================
# 5. INSTALL RULES
# ==========================================
install(TARGETS vision_node
  DESTINATION lib/${PROJECT_NAME})

# Install non-compiled files (launch, config, models) to the share directory
# This ensures 'ros2 launch' can find your YAML and AI models
install(DIRECTORY 
  launch
  config
  # model
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
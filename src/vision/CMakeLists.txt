cmake_minimum_required(VERSION 3.10)
project(vision)

# Compile using C++17
set(CMAKE_CXX_STANDARD 17)

# ==========================================
# 1. CORE DEPENDENCIES (Required)
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV 4.5 REQUIRED)

# Added for translating ROS 2 camera topics to OpenCV matrices
find_package(sensor_msgs REQUIRED) 
find_package(cv_bridge REQUIRED)   

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

# Only give these headers to vision_node. Nothing else.
target_include_directories(vision_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
)

# Link OpenCV
target_link_libraries(vision_node ${OpenCV_LIBRARIES})

# Link ROS 2 packages
ament_target_dependencies(vision_node 
  rclcpp
  sensor_msgs
  cv_bridge
)

# ==========================================
# 4. APPLY THE GPU SHIELD
# ==========================================
# If CMake finds TensorRT on the machine (like on the Jetson), it activates it.
# If not (like on your laptop), it safely skips this block.
if(CUDA_FOUND AND TENSORRT_LIBRARY AND TENSORRT_INCLUDE_DIR)
    message(STATUS "TensorRT found! Enabling GPU acceleration support.")
    
    # This creates the #ifdef USE_TENSORRT flag in your C++ code
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
install(DIRECTORY 
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

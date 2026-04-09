cmake_minimum_required(VERSION 3.10)
project(vision)

# --- Generate compile_commands.json for VS Code ---
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)

# ==========================================
# 1. HARDWARE / PROTOTYPING OPTIONS
# ==========================================
# This build is configured to use CUDA and TensorRT for inference.
message(STATUS "Building with CUDA/TensorRT support.")
set(USE_TENSORRT ON CACHE INTERNAL "Force using TensorRT for all sub-projects")

find_package(CUDA REQUIRED)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS
    /usr/include/aarch64-linux-gnu
    /usr/local/cuda/include
    $ENV{TENSORRT_DIR}/include
)
find_library(TENSORRT_LIBRARY nvinfer
  HINTS
    /usr/lib/aarch64-linux-gnu
    /usr/local/cuda/lib64
    $ENV{TENSORRT_DIR}/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT not found. Please ensure it is installed correctly, or that the TENSORRT_DIR environment variable is set.")
endif()

# ==========================================
# 2. EXTERNAL DEPENDENCIES (ROS 2 & Vision)
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

find_package(cv_bridge REQUIRED) # make sure it comes before OpenCV
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(pcl_conversions REQUIRED)

# [K1DIY] We UNCOMMENTED this because ProcessData now uses Detections.msg!
find_package(vision_interface REQUIRED)

# ==========================================
# 3. INCLUDE DIRECTORIES
# ==========================================
include_directories(
    include
    ${PROJECT_SOURCE_DIR}/thirdparty/include
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}            # <-- ADDED: CMake's detected CUDA path
    # /usr/local/cuda/include        # <-- ADDED: Common CUDA include path (adjust if your CUDA is installed elsewhere)
)

# ==========================================
# 4. DELEGATE TO THE SOURCE FOLDER
# ==========================================
add_subdirectory(src)

# ==========================================
# 5. INSTALL & EXPORT
# ==========================================
install(DIRECTORY launch config model DESTINATION share/${PROJECT_NAME})
ament_export_include_directories(include)
ament_package()
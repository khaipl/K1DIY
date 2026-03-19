cmake_minimum_required(VERSION 3.10)
project(vision)

# --- Generate compile_commands.json for VS Code ---
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)

# ==========================================
# 1. EXTERNAL DEPENDENCIES (The "Environment")
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV 4.5 REQUIRED)
find_package(sensor_msgs REQUIRED) 
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(yaml-cpp REQUIRED) 
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

# [NOTE] Commented out until you build your interface module
# find_package(vision_interface REQUIRED)

# ==========================================
# 2. INCLUDE DIRECTORIES
# ==========================================
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
)

# ==========================================
# 3. DELEGATE TO THE SOURCE FOLDER
# ==========================================
add_subdirectory(src)

# ==========================================
# 4. INSTALL & EXPORT
# ==========================================
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
ament_export_include_directories(include)
ament_package()
# **K1DIY \- Team Naova RoboCup Vision Stack (Local Debugging)**

Welcome to the K1DIY vision stack for the Booster K1 humanoid robot. This project adapts the official Booster SDK for autonomous RoboCup soccer, featuring a highly optimized, Sim2Real-capable ROS 2 architecture.  
This guide specifically covers how to build and test the vision\_node locally on a non-GPU laptop before deploying it to the Jetson-powered robot.

## **Key Features (Vision Node)**

* Hardware-Agnostic AI: Uses a Strategy Pattern to run YOLOv8 via ONNX (CPU) on laptops, automatically switching to TensorRT (GPU) when compiled on the robot.  
* Dual-Stream Synchronization: Subscribes to and synchronizes both color (left eye) and depth (StereoNet) image streams.  
* Auto-Recording: Automatically saves 5 seconds of synchronized raw, edge-detected, and depth-colorized .avi video to K1DIY/data/test/ on startup for Sim2Real validation.  
* Safe Mode: Toggle enable\_ai: false in vision.yaml to bypass heavy inference and test camera pipelines and edge detection seamlessly.

## **Laptop Debugging Workflow**

To test the vision pipeline on your local machine without the physical robot hardware, open three separate terminals and run the following commands.

### **Terminal 1: Build and Launch the Vision Node**

Compile the ROS 2 workspace (the CMake script will automatically detect the absence of TensorRT and build the CPU fallback) and launch the node.  

```bash
# Navigate to your workspace root (e.g., ~/NaovaCodeK1)
./scripts/build.sh
source install/setup.bash
ros2 launch vision launch.py
```

### **Terminal 2: Visualize the Output**

The vision\_node processes the camera feed (applying Canny edge detection) and publishes a lightweight mono8 debug image. Use rqt\_image\_view to see what the robot's "brain" is seeing in real-time.  

```bash
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view
```

Tip: In the rqt\_image\_view GUI, select the /vision/debug\_edges topic from the dropdown menu.

### **Terminal 3 (Optional): Feed Live Laptop Camera Data**

If you don't have a rosbag playing recorded robot data, you can use your laptop's built-in webcam to feed live images to the network.  

```bash
source /opt/ros/humble/setup.bash
ros2 run v4l2_camera v4l2_camera_node --ros-args -p video_device:="/dev/video0" -r /image_raw:=/booster_camera_bridge/image_left_raw
```

Developer Note on Topics: By default, vision.yaml expects the camera stream on /booster\_camera\_bridge/image\_left\_raw. If you are using your laptop camera via the command above, you will need to either update camera\_topic in src/vision/config/vision.yaml to match your webcam's topic (/camera/image\_raw), or adjust the remap argument in Terminal 3 to: \-r /image\_raw:=/booster\_camera\_bridge/image\_left\_raw.

## **Configuration**

Before running, ensure your src/vision/config/vision.yaml is set up for local laptop testing:

* enable\_ai: false (or true if you want to test the ONNX model).  
* backend: "cpu\_onnx"  
* use\_depth: false (unless you are mocking the depth topic locally, as standard webcams do not provide stereo depth).
# Local Debugging
## The Depth Synchronization Requirement

The `vision_node` uses a `message_filters::Synchronizer` with an `ApproximateTime` policy. This means the C++ node strictly waits for **both** a color frame and a depth frame with closely matching timestamps before executing any processing (like edge detection or AI inference). 

Standard laptop webcams only provide color images. To prevent the node from starving while waiting for depth data, you must run a mock depth publisher that generates fake 16-bit depth frames stamped with the exact same timestamp as your webcam frames.

## Laptop Debugging Workflow

To test the vision pipeline on your local machine, open four separate terminals and run the following commands in order.
### Terminal 1: Feed Live Laptop Camera Data

Capture your laptop's built-in webcam and publish it to the expected hardware topic.

```bash
source /opt/ros/humble/setup.bash
ros2 run v4l2_camera v4l2_camera_node --ros-args -p video_device:="/dev/video0" -r /image_raw:=/booster_camera_bridge/image_left_raw
```

### Terminal 2: Run the Depth Mock Node

Tricks the C++ node into believing the stereo hardware is active.

```bash
source install/setup.bash
python3 src/vision/scripts/depth_mock.py
```

### Terminal 3: Build and Launch the Vision Node

Compile the ROS 2 workspace (the CMake script will automatically detect the absence of TensorRT and build the CPU fallback) and launch the node.

```bash
# Navigate to your workspace root (e.g., ~/NaovaCodeK1)
./scripts/build.sh
source install/setup.bash
ros2 launch vision launch.py
```

Terminal 4: Visualize the Output

The `vision_node` will process the synchronized camera feeds and publish a lightweight `mono8` debug image.

```bash
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view
```

## Configuration

Before running, ensure your src/vision/config/vision.yaml is set up for local laptop testing:

* `enable\ai: false` (or `true` if you want to test the ONNX model).  
* `backend: "cpu\onnx"`
* `use\depth: false` (unless you are mocking the depth topic locally, as standard webcams do not provide stereo depth).
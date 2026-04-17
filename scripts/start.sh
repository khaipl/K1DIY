#!/bin/bash

# Master script to run the robot fully autonomously in the background
cd `dirname $0`
cd ..

echo "[K1DIY] INITIALIZING AUTONOMOUS STACK (Background Mode)"
./scripts/stop.sh

source /opt/ros/humble/setup.bash
source ./install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/booster/BoosterRos2/fastdds_profile_udp_only.xml

ros2 daemon stop > /dev/null 2>&1
ros2 daemon start

echo "-> Starting VISION..."
nohup ros2 launch vision launch.py use_sim_time:=false > vision.log 2>&1 &
VISION_PID=$!

echo "-> Waiting 10s for ONNX/TensorRT warmup..."
sleep 10

# Verification: Is Vision still running?
if ! kill -0 $VISION_PID 2>/dev/null; then
    echo "[CRITICAL] Vision node failed to start. Check vision.log"
    exit 1
fi

echo "-> Starting BRAIN..."
nohup ros2 launch brain launch.py use_sim_time:=false "$@" > brain.log 2>&1 &

echo "-> Starting GAME_CONTROLLER..."
nohup ros2 launch game_controller launch.py > game_controller.log 2>&1 &

echo "[SUCCESS] K1DIY is now running in the background."
echo "------------------------------------------------"
echo "To watch the Brain:   tail -f brain.log"
echo "To watch the Vision:  tail -f vision.log"
echo "To stop everything:   ./scripts/stop.sh"
echo "------------------------------------------------"

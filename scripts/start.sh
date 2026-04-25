#!/bin/bash

# Move to workspace root
cd `dirname $0`
cd ..

echo "[K1DIY] INITIALIZING AUTONOMOUS STACK (Background Mode)"
# Stop existing nodes before starting new ones to avoid conflicts
./scripts/stop.sh

source /opt/ros/humble/setup.bash
source ./install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/booster/BoosterRos2/fastdds_profile_udp_only.xml

# Generate a unique timestamp for this run (Format: YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_BASE_DIR="log"
CURRENT_LOG_DIR="${LOG_BASE_DIR}/run_${TIMESTAMP}"

# Create the directories if they don't exist
mkdir -p "${CURRENT_LOG_DIR}"

echo "-> Logs for this session will be saved in: ${CURRENT_LOG_DIR}"
echo "-> Starting Discovery Daemon..."
ros2 daemon stop > /dev/null 2>&1
ros2 daemon start

echo "-> Launching VISION (Background)..."
# Redirect output to the timestamped vision log
nohup ros2 launch vision launch.py use_sim_time:=false > "${CURRENT_LOG_DIR}/vision.log" 2>&1 &
VISION_PID=$!

# Wait for ONNX/TensorRT to initialize in GPU memory
echo "-> Waiting 3s for Vision to initialize..."
sleep 3

# Verify Vision is still alive before starting the brain
if ! kill -0 $VISION_PID 2>/dev/null; then
    echo "[ERROR] Vision node died immediately! Check ${CURRENT_LOG_DIR}/vision.log"
    exit 1
fi

echo "-> Launching BRAIN (Background)..."
# The brain executable handles both main logic and high-priority callbacks via threads
nohup ros2 launch brain launch.py use_sim_time:=false "$@" > "${CURRENT_LOG_DIR}/brain.log" 2>&1 &

echo "-> Launching GAME CONTROLLER (Background)..."
nohup ros2 launch game_controller launch.py > "${CURRENT_LOG_DIR}/game_controller.log" 2>&1 &

# Update a symlink called 'latest' so you can always find the newest logs easily
ln -sfn "run_${TIMESTAMP}" "${LOG_BASE_DIR}/latest"

echo "[SUCCESS] K1DIY is now autonomous."
echo "------------------------------------------------"
echo "Watch Live Brain:   tail -f ${LOG_BASE_DIR}/latest/brain.log"
echo "Watch Live Vision:  tail -f ${LOG_BASE_DIR}/latest/vision.log"
echo "------------------------------------------------"

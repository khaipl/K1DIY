#!/bin/bash

# Move to workspace root
cd `dirname $0`
cd ..

echo "[K1DIY] Starting Vision only..."
./scripts/stop.sh

# Source environment
source /opt/ros/humble/setup.bash
source ./install/setup.bash

# Ensure we are using the correct DDS profile for the Jetson
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/booster/BoosterRos2/fastdds_profile_udp_only.xml

ros2 daemon stop > /dev/null 2>&1
ros2 daemon start

# use_sim_time:=false for the physical K1 robot
# use_sim_time:=true for Webots/Isaac Sim
ros2 launch vision launch.py use_sim_time:=false "$@"

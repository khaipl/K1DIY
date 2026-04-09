#!/bin/bash

# 1. Give the background script access to the ROS 2 Network!
source /opt/ros/humble/setup.bash
source ~/K1DIY/install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/booster/BoosterRos2/fastdds_profile_udp_only.xml

# 2. Point exactly to the sandboxed Naova Webots
export WEBOTS_HOME=$HOME'/K1DIY/simulation/webots'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME'/lib/controller'

echo "Starting Webots World..."
$WEBOTS_HOME/webots ~/K1DIY/simulation/K1_v1.wbt &

# 3. Give Webots a head start to open its communication ports
sleep 3

echo "Starting Booster Runner (The Bridge)..."
~/K1DIY/simulation/booster-runner-full-webots-k1-0.0.1.run &

wait
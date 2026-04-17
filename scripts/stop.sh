#!/bin/bash

echo "[K1DIY] Stopping all Robocup nodes..."

# Kill Vision Node
echo "-> Stopping Vision..."
pkill -9 vision_node > /dev/null 2>&1

# Kill Python helpers
echo "-> Stopping Python converters..."
pkill -9 -f detection_converter_node.py > /dev/null 2>&1

# Kill Brain Node
echo "-> Stopping Brain (Behavior Tree)..."
pkill -9 brain_node > /dev/null 2>&1

# Kill Game Controller
echo "-> Stopping Game Controller..."
ps aux | grep "game_controller" | grep -v "game_controller_app" | grep -v "grep" | awk '{print $2}' | xargs -r kill -9 > /dev/null 2>&1

# Cleanup ROS 2 Daemon
echo "-> Cleaning up ROS 2 daemon..."
ros2 daemon stop > /dev/null 2>&1

echo "[DONE] All nodes stopped."

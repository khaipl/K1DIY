#!/bin/bash

# Navigate to the project root from the scripts folder 
cd `dirname $0`
cd ..

# Source ROS 2 Humble environment
source /opt/ros/humble/setup.bash

export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning,ignore::FutureWarning"

# [K1DIY] Build the workspace. 
# Defaults to whatever the CMakeLists.txt defaults to (currently NO_CUDA=ON for prototyping).
# Any arguments passed to this script (like --cmake-args) are appended to the colcon command.
colcon build --symlink-install --base-paths src "$@"

# Auditory feedback for the developer
espeak "build complete" >/dev/null 2>&1 || echo "Build complete"
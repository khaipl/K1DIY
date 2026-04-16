#!/bin/bash

# Navigate to the scripts directory 
cd `dirname $0`

# [K1DIY] Explicitly trigger the NO_CUDA path in our merged CMake logic.
# Added -O3 for performance optimization on non-GPU CPUs.
./build.sh --cmake-args \
    -DNO_CUDA=ON \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS="-g -rdynamic -O3"
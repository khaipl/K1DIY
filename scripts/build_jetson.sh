#!/bin/bash

# Navigate to the scripts directory 
cd `dirname $0`

# [K1DIY] Explicitly enable CUDA/TensorRT for the Jetson Orin architecture.
./build.sh --cmake-args \
    -DNO_CUDA=OFF \
    -DCMAKE_BUILD_TYPE=Release
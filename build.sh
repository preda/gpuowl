#!/bin/sh

#update the path to libOpenCL in -L below, as needed.
g++ -O2 -std=c++11 gpuowl.cpp -ogpuowl -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -lOpenCL

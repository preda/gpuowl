# gpuOwL
GPU Lucas-Lehmer primality test.

## Build
Use either build.sh or make.

### Prerequisites:
* a C++ compiler (e.g. gcc, clang)
* an OpenCL implementation (which provides the **libOpenCL** library), e.g. AMDGPU-Pro

The build is just a C++ compiler invocation on gpuowl.cpp, specifying the OpenCL library in the -L to the compiler, e.g.:

g++ gpuowl.cpp -ogpuowl -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -lOpenCL

## Invocation
gpuowl <exponent>, e.g. gpuowl 39527687, to perform primality test of 2^exponent - 1.


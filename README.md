# gpuOwL
GPU Lucas-Lehmer primality test.

## Build
Use make or build.sh.

### Prerequisites:
* a C++ compiler (e.g. gcc, clang)
* an OpenCL implementation (which provides the **libOpenCL** library), e.g. AMDGPU-Pro

The build is just a C++ compiler invocation on gpuowl.cpp, specifying the OpenCL library in the -L to the compiler, e.g.:

g++ gpuowl.cpp -ogpuowl -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -lOpenCL

## Usage
* Get exponents for testing from GIPMS Manual Testing ( http://mersenne.org/ ). gpuOwL best handles exponents is the vicinity of 77M, less than 78M.
* Copy the lines from GIMPS to a file named 'worktodo.txt'
* Run gpuowl. It prints progress report on stdout and in gpuowl.log, and writes result lines to results.txt
* Submit the result lines from results.txt to http://mersenne.org/ manual testing.

## Command line arguments
* It is possible to pass arguments to the OpenCL compiler by passing "-cl" to gpuowl. For example, to dump the compiled ISA use
**gpuowl -cl -save-temps** ; to put the ISA into an *existing* folder 'isa' use **gpuowl -cl -save-temps=isa/**

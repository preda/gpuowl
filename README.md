# gpuOWL
GPU Lucas-Lehmer primality test.

## Build
Use make or build.sh.

### Prerequisites:
* a C++ compiler (e.g. gcc, clang)
* an OpenCL implementation (which provides the **libOpenCL** library), e.g. AMDGPU-Pro

The build is just a C++ compiler invocation on gpuowl.cpp, specifying the OpenCL library in the -L to the compiler, e.g.:

g++ gpuowl.cpp -ogpuowl -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -lOpenCL

## Usage
* Get exponents for testing from GIPMS, Manual Testing ( http://mersenne.org/ ). gpuOWL best handles exponents is the vecinity of 77M, less than 78M.
* Copy the lines from GIMPS to a file named 'worktodo.txt'
* Run gpuowl. It will print progress report on stdout and in gpuowl.log, and will write result lines to results.txt
* Submit the result lines from results.txt to http://mersenne.org/ manual testing.

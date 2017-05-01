# Edit the path in -L below if needed, to the folder containing OpenCL.dll on Windows or libOpenCL.so on UNIX.
# The included paths are for MSYS-2/Windows and AMDGPU-Pro/Linux.
# -march=native produces non-portable executables (that are guranteed to run only no this machine).
# -mtune=native

gpuowl: gpuowl.cpp clwrap.h tinycl.h
	g++ -O2 -Wall -Werror -std=c++14 gpuowl.cpp -ogpuowl -lOpenCL -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -L/c/Windows/System32



# Edit the path in -L below if needed, to the folder containing OpenCL.dll on Windows or libOpenCL.so on UNIX.
# The included paths are for MSYS-2/Windows and AMDGPU-Pro/Linux.

gpuowl: gpuowl.cpp args.h checkpoint.h clwrap.h common.h timeutil.h tinycl.h worktodo.h
	g++ -O2 -Wall -Werror -std=c++14 gpuowl.cpp -ogpuowl -lOpenCL -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -L/c/Windows/System32

debug: gpuowl.cpp args.h checkpoint.h clwrap.h common.h timeutil.h tinycl.h worktodo.h
	g++ -g -Wall -Werror -std=c++14 gpuowl.cpp -oowldebug -lOpenCL -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -L/c/Windows/System32

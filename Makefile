# Edit the path in -L below if needed, to the folder containing OpenCL.dll on Windows or libOpenCL.so on UNIX.
# The included lib paths are for ROCm, AMDGPU-pro/Linux or MSYS-2/Windows.

HEADERS = args.h checkpoint.h clwrap.h common.h timeutil.h tinycl.h worktodo.h nttshared.h
LIBPATH = -L/opt/rocm/opencl/lib/x86_64 -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -L/c/Windows/System32

gpuowl: gpuowl.cpp ${HEADERS}
	g++ -O2 -Wall -Werror -std=c++14 gpuowl.cpp -ogpuowl -lOpenCL ${LIBPATH}

debug: gpuowl.cpp ${HEADERS}
	g++ -g -Wall -Werror -std=c++14 gpuowl.cpp -odebug -lOpenCL ${LIBPATH}

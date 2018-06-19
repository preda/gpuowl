# Edit the path in -L below if needed, to the folder containing OpenCL.dll on Windows or libOpenCL.so on UNIX.
# The included lib paths are for ROCm, AMDGPU-pro/Linux or MSYS-2/Windows.

HEADERS = args.h clwrap.h common.h kernel.h state.h stats.h timeutil.h tinycl.h worktodo.h
SRCS = Gpu.cpp common.cpp
LIBPATH = -L/opt/rocm/opencl/lib/x86_64 -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -L/c/Windows/System32

all: gpuowl cudaowl fftbench

gpuowl: gpuowl.cpp ${HEADERS} ${SRCS}
	g++ -DREV=\"`git rev-parse --short HEAD``git diff-files --quiet || echo -mod`\" -O2 -Wall -Werror -std=c++14 gpuowl.cpp ${SRCS} -ogpuowl -lOpenCL ${LIBPATH}

cudaowl: cudaowl.cu ${SRCS}
	nvcc -O2 -o cudaowl cudaowl.cu ${SRCS} -lcufft

fftbench: fftbench.cu
	nvcc -O2 -o fftbench fftbench.cu -lcufft

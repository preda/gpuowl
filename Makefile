# Choose one of: openowl (OpenCL) or cudaowl (CUDA).
#all: openowl cudaowl

HEADERS = args.h clwrap.h common.h kernel.h state.h stats.h timeutil.h tinycl.h worktodo.h Gpu.h LowGpu.h 
SRCS = Gpu.cpp common.cpp gpuowl.cpp

# Edit the path in -L below if needed, to the folder containing OpenCL.dll on Windows or libOpenCL.so on UNIX.
# The included lib paths are for ROCm, AMDGPU-pro/Linux or MSYS-2/Windows.
LIBPATH = -L/opt/rocm/opencl/lib/x86_64 -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -L/c/Windows/System32

openowl: ${HEADERS} ${SRCS} OpenGpu.h OpenGpu.cpp
	g++ -DREV=\"`git rev-parse --short HEAD``git diff-files --quiet || echo -mod`\" -Wall -Werror -std=c++14 OpenGpu.cpp ${SRCS} -o openowl -lOpenCL ${LIBPATH}

cudaowl: ${HEADERS} ${SRCS} CudaGpu.h CudaGpu.cu
	nvcc -DREV=\"`git rev-parse --short HEAD``git diff-files --quiet || echo -mod`\" -o cudaowl CudaGpu.cu ${SRCS} -lcufft

#fftbench: fftbench.cu
#	nvcc -O2 -o fftbench fftbench.cu -lcufft

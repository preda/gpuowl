HEADERS = args.h checkpoint.h clwrap.h common.h kernel.h state.h stats.h timeutil.h tinycl.h worktodo.h Gpu.h LowGpu.h TF.h OpenTF.h ghzdays.h OpenGpu.h
SRCS = Worktodo.cpp Result.cpp common.cpp gpuowl.cpp Gpu.cpp OpenGpu.cpp clwrap.cpp Task.cpp checkpoint.cpp timeutil.cpp

# Edit the path in -L below if needed, to the folder containing OpenCL.dll on Windows or libOpenCL.so on UNIX.
# The included lib paths are for ROCm, AMDGPU-pro/Linux or MSYS-2/Windows.
LIBPATH = -L/opt/rocm/opencl/lib/x86_64 -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -L/c/Windows/System32 -L.

openowl: ${HEADERS} ${SRCS}
	g++ -O2 -DREV=\"`git rev-parse --short HEAD``git diff-files --quiet || echo -mod`\" -Wall -std=c++14 ${SRCS} -o openowl -lOpenCL -lgmp ${LIBPATH}

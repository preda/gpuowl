HEADERS = Background.h Pm1Plan.h GmpUtil.h Args.h checkpoint.h clwrap.h common.h kernel.h state.h timeutil.h tinycl.h Worktodo.h Gpu.h Primes.h Signal.h FFTConfig.h
SRCS = Pm1Plan.cpp GmpUtil.cpp Worktodo.cpp common.cpp gpuowl.cpp Gpu.cpp clwrap.cpp Task.cpp checkpoint.cpp timeutil.cpp Args.cpp Primes.cpp state.cpp Signal.cpp FFTConfig.cpp

# Edit the path in -L below if needed, to the folder containing OpenCL.dll on Windows or libOpenCL.so on UNIX.
# The included lib paths are for ROCm, AMDGPU-pro/Linux or MSYS-2/Windows.
LIBPATH = -L/opt/rocm/opencl/lib/x86_64 -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -L/c/Windows/System32 -L.

#-fsanitize=leak

openowl: ${HEADERS} ${SRCS}
	g++ -Wall -O2 -std=c++17 -DREV=\"`git rev-parse --short HEAD``git diff-files --quiet || echo -mod`\" -Wall ${SRCS} -o openowl -lOpenCL -lgmp -lstdc++fs -pthread ${LIBPATH}

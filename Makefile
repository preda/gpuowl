# Edit the path in -L below if needed, to the folder containing OpenCL.dll on Windows or libOpenCL.so on UNIX.
# The included paths are for MSYS-2/Windows and AMDGPU-Pro/Linux.

# The build now needs GMP https://gmplib.org/ for the Jacobi-symbol check (a good thing).
# To compile without GMP (and disable the check), define NO_GMP and may drop -lgmp below.

gpuowl: gpuowl.cpp args.h checkpoint.h clwrap.h common.h timeutil.h tinycl.h worktodo.h
	g++ -O2 -Wall -Werror -std=c++14 gpuowl.cpp -ogpuowl -lgmp -lOpenCL -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -L/c/Windows/System32

# Edit the path in -L below if needed, to the folder containing OpenCL.dll on Windows or libOpenCL.so on UNIX.
# The included paths are for MSYS-2/Windows and AMDGPU-Pro/Linux.

# Define JACOBI to check the Jacobi symbol. This recquires GMP (GNU MP).
# invoke e.g.: rm ./gpuowl ; make JACOBI=1
# or export JACOBI=1 ; make

ifdef JACOBI
JAC=-DJACOBI -lgmp
endif

gpuowl: gpuowl.cpp args.h checkpoint.h clwrap.h common.h timeutil.h tinycl.h worktodo.h
	g++ -O2 -Wall -Werror -std=c++14 gpuowl.cpp -ogpuowl $(JAC) -lOpenCL -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -L/c/Windows/System32

DefaultEnvironment(CXX='g++-9')

srcs = 'clpp.cpp Pm1Plan.cpp GmpUtil.cpp FFTConfig.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp checkpoint.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp'.split()

AlwaysBuild(Command('version.inc', [], 'echo \\"`git describe --long --dirty --always`\\" > $TARGETS'))
AlwaysBuild(Command('gpuowl-wrap.cl', ['gpuowl.cl'], 'cat head.txt gpuowl.cl tail.txt > gpuowl-wrap.cl'))

LIBPATH=['/opt/rocm/opencl/lib/x86_64']

Program('gpuowl', srcs, LIBPATH=LIBPATH, LIBS=['amdocl64', 'gmp', 'stdc++fs'], parse_flags='-std=c++17 -O2 -Wall -pthread')

# Program('asm', 'asm.cpp clpp.cpp clwrap.cpp'.split(), LIBS=['OpenCL'], parse_flags='-std=c++17 -O2 -Wall -pthread')

#DefaultEnvironment(CXX='g++-9')

srcs = 'Pm1Plan.cpp GmpUtil.cpp FFTConfig.cpp Worktodo.cpp common.cpp gpuowl.cpp Gpu.cpp clwrap.cpp Task.cpp checkpoint.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp'.split()

AlwaysBuild(Command('version.inc', [], 'echo \\"`git describe --long --dirty --always`\\" > $TARGETS'))

LIBPATH=['/opt/rocm/opencl/lib/x86_64']

Program('gpuowl', srcs, LIBPATH=LIBPATH, LIBS=['amdocl64', 'gmp', 'stdc++fs'], parse_flags='-std=c++17 -O2 -Wall -pthread')

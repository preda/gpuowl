import os

env = Environment(CXX='g++-9')
env['ENV']['TERM'] = os.environ['TERM']

#DefaultEnvironment(CXX='g++-9')

srcs = 'clpp.cpp Pm1Plan.cpp GmpUtil.cpp FFTConfig.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp checkpoint.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp gpuowl-wrap.cpp'.split()

AlwaysBuild(Command('version.inc', [], 'echo \\"`git describe --long --dirty --always`\\" > $TARGETS'))
AlwaysBuild(Command('gpuowl-wrap.cpp', ['gpuowl.cl'], 'cat head.txt gpuowl.cl tail.txt > gpuowl-wrap.cpp'))

LIBPATH=['/opt/rocm/opencl/lib/x86_64']

env.Program('gpuowl', srcs, LIBPATH=LIBPATH, LIBS=['amdocl64', 'gmp', 'stdc++fs'], parse_flags='-std=c++17 -O2 -Wall -pthread -fdiagnostics-color=auto -fmax-errors=6')

# Program('asm', 'asm.cpp clpp.cpp clwrap.cpp'.split(), LIBS=['OpenCL'], parse_flags='-std=c++17 -O2 -Wall -pthread')

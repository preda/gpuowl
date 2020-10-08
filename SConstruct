import os

env = Environment()
#CXX='g++-9')
env['ENV']['TERM'] = os.environ['TERM']

#DefaultEnvironment(CXX='g++-9')

srcs = 'B1Accumulator.cpp Memlock.cpp log.cpp md5.cpp sha3.cpp AllocTrac.cpp GmpUtil.cpp FFTConfig.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp checkpoint.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp gpuowl-wrap.cpp'.split()

AlwaysBuild(Command('version.inc', [], 'echo \\"`git describe --tags --long --dirty --always`\\" > $TARGETS'))
AlwaysBuild(Command('gpuowl-expanded.cl', ['gpuowl.cl'], './tools/expand.py < gpuowl.cl > gpuowl-expanded.cl'))
AlwaysBuild(Command('gpuowl-wrap.cpp', ['gpuowl-expanded.cl'], 'cat head.txt gpuowl-expanded.cl tail.txt > gpuowl-wrap.cpp'))

LIBPATH=['/opt/rocm-3.3.0/opencl/lib/x86_64', '/opt/rocm-3.5.0/opencl/lib']

config = '-g'
#config = '-g -O2'
#-fsanitize=address'
# -fstack-protector-strong -static-libasan'
#config = '-O2'

env.Program('gpuowl', srcs, LIBPATH=LIBPATH, LIBS=['amdocl64', 'gmp', 'stdc++fs'], parse_flags='-std=c++17 -Wall -pthread ' + config)
#'amdocl64'

# Program('asm', 'asm.cpp clpp.cpp clwrap.cpp'.split(), LIBS=['OpenCL'], parse_flags='-std=c++17 -O2 -Wall -pthread')

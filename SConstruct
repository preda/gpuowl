import os

env = Environment(CXX='g++-10')
env['ENV']['TERM'] = os.environ['TERM']

# DefaultEnvironment(CXX='g++-10')

srcs = 'ProofCache.cpp Proof.cpp Pm1Plan.cpp B1Accumulator.cpp Memlock.cpp log.cpp md5.cpp sha3.cpp AllocTrac.cpp GmpUtil.cpp FFTConfig.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp Saver.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp gpuowl-wrap.cpp'.split()

AlwaysBuild(Command('version.inc', [], 'echo \\"`git describe --tags --long --dirty --always`\\" > $TARGETS'))
AlwaysBuild(Command('gpuowl-expanded.cl', ['gpuowl.cl'], './tools/expand.py < gpuowl.cl > gpuowl-expanded.cl'))
AlwaysBuild(Command('gpuowl-wrap.cpp', ['gpuowl-expanded.cl'], 'cat head.txt gpuowl-expanded.cl tail.txt > gpuowl-wrap.cpp'))

LIBPATH=['/opt/rocm/opencl/lib', '/opt/rocm-3.3.0/opencl/lib/x86_64', '/opt/rocm-3.5.0/opencl/lib']

#config = '-g'
config = '-g -O2'
#-fsanitize=address'
# -fstack-protector-strong -static-libasan'
#config = '-O2'

flags = '-std=c++17 -Wall -pthread ' + config
env.Program('gpuowl', srcs, LIBPATH=LIBPATH, LIBS=['amdocl64', 'gmp', 'stdc++fs'], parse_flags=flags)
# env.Program('D', ['D.cpp', 'Pm1Plan.cpp', 'log.cpp', 'common.cpp', 'timeutil.cpp'], parse_flags=flags)

# Program('asm', 'asm.cpp clpp.cpp clwrap.cpp'.split(), LIBS=['OpenCL'], parse_flags='-std=c++17 -O2 -Wall -pthread')

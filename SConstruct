srcs = 'GmpUtil.cpp FFTConfig.cpp Worktodo.cpp Result.cpp common.cpp gpuowl.cpp Gpu.cpp clwrap.cpp Task.cpp checkpoint.cpp timeutil.cpp Args.cpp Primes.cpp state.cpp Signal.cpp'.split()

Program('openowl', srcs, LIBPATH='.', LIBS=['amdocl64', 'gmp'], parse_flags='-std=c++17 -O2 -Wall')

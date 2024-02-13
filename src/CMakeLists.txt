
add_executable(gpuowl clbundle.cpp ProofCache.cpp Proof.cpp
  Memlock.cpp log.cpp md5.cpp sha3.cpp AllocTrac.cpp FFTConfig.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp Saver.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp
  File.cpp
  gpuid.h
  gpuid.cpp
  version.cpp)

target_link_libraries(gpuowl gmp OpenCL)


add_executable(gpuowl bundle.cpp ProofCache.cpp Proof.cpp
  Memlock.cpp log.cpp md5.cpp sha3.cpp AllocTrac.cpp FFTConfig.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp
  File.cpp
  gpuid.cpp
  version.cpp
  bundle.cpp
  KernelCompiler.cpp
  Kernel.cpp
  SaveMan.cpp
  Saver.cpp
  Queue.cpp
  TimeInfo.cpp
  Profile.cpp
  Event.cpp
  TrigBufCache.cpp
  )

target_link_libraries(gpuowl OpenCL)

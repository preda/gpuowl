
add_executable(gpuowl bundle.cpp ProofCache.cpp Proof.cpp
  Memlock.cpp log.cpp md5.cpp sha3.cpp AllocTrac.cpp FFTConfig.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp Saver.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp
  File.cpp
  gpuid.h
  gpuid.cpp
  version.cpp
  bundle.h
  bundleImpl.cpp
  KernelCompiler.h
  KernelCompiler.cpp
  cl/ffthin.cl
  cl/ffthout.cl
  cl/etc.cl
  cl/fftMiddleIn.cl
  cl/fftmiddleout.cl
  cl/tailfusedmul.cl
  cl/tailsquare.cl
  cl/multiply.cl
  cl/carry.cl
  cl/carryb.cl
  )

target_link_libraries(gpuowl gmp OpenCL)

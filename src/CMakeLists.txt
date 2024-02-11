
# file(GLOB opencl_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cl)

#add_custom_command(OUTPUT clbundle.cpp COMMAND expand ${CMAKE_CURRENT_SOURCE_DIR}/gpuowl.cl clbundle.cpp
#                   DEPENDS ${opencl_files} expand)

add_executable(gpuowl clbundle.cpp ProofCache.cpp Proof.cpp Memlock.cpp log.cpp md5.cpp sha3.cpp AllocTrac.cpp GmpUtil.cpp FFTConfig.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp Saver.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp
  File.cpp
  gpuid.h
  gpuid.cpp)




# find_library(AMDOCL amdocl64 /opt/rocm/lib)
# link_directories(/opt/rocm/lib)
# add_library(amdocl SHARED IMPORTED)
# set_target_properties(amdocl PROPERTIES IMPORTED_LOCATION "/opt/rocm/lib/libamdocl64.so")

# target_link_directories(gpuowl PRIVATE /opt/rocm/lib)
target_link_libraries(gpuowl gmp OpenCL)
# amdocl64)

# configure_file(version.h.in version.h)

#target_include_directories(gpuowl PUBLIC
#  "${PROJECT_BINARY_DIR}"
#  "${PROJECT_SOURCE_DIR}/src"
#  )


add_executable(prpll
  Trig.cpp
  Primes.cpp
  bundle.cpp
  Proof.cpp
  log.cpp md5.cpp sha3.cpp AllocTrac.cpp FFTConfig.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp
  File.cpp
  gpuid.cpp
  version.cpp
  KernelCompiler.cpp
  Kernel.cpp
  Saver.cpp
  Queue.cpp
  TimeInfo.cpp
  Profile.cpp
  Event.cpp
  TrigBufCache.cpp
  CycleFile.cpp
  tune.cpp
  TuneEntry.cpp
  fs.cpp
  version.inc
  )

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # Real GCC (not clang), needed for 128-bit floats and std::filesystem::path
  set(CMAKE_CXX_COMPILER g++-14)
endif()

find_package(OpenCL)

if (${OpenCL_FOUND})
  # Use OpenCL library found by cmake
  target_link_libraries(prpll OpenCL::OpenCL)
else()
  # Pass -lOpenCL to the linker and hope for the best
  target_link_libraries(prpll OpenCL)
endif()

target_include_directories(prpll PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_custom_command(
  OUTPUT version.inc
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND git describe --tags --long --dirty --always --match ${CMAKE_SOURCE_DIR} | sed 's/.*/"&"/' > ${CMAKE_CURRENT_BINARY_DIR}/version.inc
  DEPENDS ${CMAKE_SOURCE_DIR}
  )

file(
  GLOB CL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cl/*.cl
  )

add_custom_command(
  OUTPUT bundle.cpp
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_SOURCE_DIR}/genbundle.sh ${CL_SOURCES} > ${CMAKE_CURRENT_BINARY_DIR}/bundle.cpp
  DEPENDS ${CL_SOURCES}
  )

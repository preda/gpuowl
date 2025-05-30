if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CMAKE_CXX_COMPILE g++-14)
endif()

#Replace add_custom_command version.inc
execute_process(
  COMMAND git describe --tags --long --dirty --always --match v/prpll/*
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

if (NOT GIT_VERSION)
  set(GIT_VERSION "unknown")
  message(WARNING "Cant take version, using default value")
endif()

set(GIT_VERSION_QUOTED "\"${GIT_VERSION}\"")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/version.inc" "${GIT_VERSION_QUOTED}\n")

file(GLOB CL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cl/*.cl")

set(bundle_cpp "${CMAKE_CURRENT_BINARY_DIR}/bundle.cpp")

add_custom_command(
  OUTPUT ${bundle_cpp}
  COMMAND ${CMAKE_COMMAND} -DCL_SOURCES="${CL_SOURCES}" -DOUTPUT=${bundle_cpp} -P ${CMAKE_SOURCE_DIR}/genbundle.cmake
  DEPENDS ${CL_SOURCES}
  COMMENT "Generation bundle.cpp"
)

add_library(bundle STATIC ${bundle_cpp})

add_executable(prpll
  Trig.cpp
  Primes.cpp
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

target_link_libraries(prpll PRIVATE bundle)

find_package(OpenCL)

if (OpenCL_FOUND)
  # Use OpenCL library found by cmake
  target_link_libraries(prpll PRIVATE OpenCL::OpenCL)
else()
  # Pass -lOpenCL to the linker and hope for the best
  target_link_libraries(prpll PRIVATE OpenCL)
endif()

target_include_directories(prpll PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
cmake_minimum_required(VERSION 3.5)

project(gpuowl LANGUAGES CXX)
# project(gpuowl VERSION 7.3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(expand IMPORTED)
set_property(TARGET expand PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/tools/expand.py)

# set_target_properties(expand PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/tools/expand.py)

add_subdirectory(src)

#ADD_CUSTOM_COMMAND(
#  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/version.cpp ${CMAKE_CURRENT_BINARY_DIR}/_version.cpp
#  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/version.cmake)

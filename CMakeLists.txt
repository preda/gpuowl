cmake_minimum_required(VERSION 3.5)

project(gpuowl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# add_executable(expand IMPORTED)
# set_property(TARGET expand PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/tools/expand.py)

add_subdirectory(src)

cmake_minimum_required(VERSION 3.13)
project(gpuowl_mirror)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

add_executable(gpuowl_mirror
        Args.cpp
        Args.h
        clwrap.cpp
        clwrap.h
        common.cpp
        common.h
        FFTConfig.cpp
        FFTConfig.h
        File.h
        GmpUtil.cpp
        GmpUtil.h
        Gpu.cpp
        Gpu.h
        gpuowl.cl
        kernel.h
        main.cpp
        primenet.py
        shared.h
        Signal.cpp
        Signal.h
        state.cpp
        state.h
        Task.cpp
        Task.h
        timeutil.cpp
        timeutil.h
        tinycl.h
        Worktodo.cpp
        Worktodo.h)

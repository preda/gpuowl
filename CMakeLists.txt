cmake_minimum_required(VERSION 3.13)
project(gpuowl_mirror)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

add_executable(gpuowl_mirror
        Args.cpp
        args.h
        checkpoint.cpp
        checkpoint.h
        clwrap.cpp
        clwrap.h
        common.cpp
        common.h
        FFTConfig.cpp
        FFTConfig.h
        file.h
        GmpUtil.cpp
        GmpUtil.h
        Gpu.cpp
        Gpu.h
        gpuowl.cl
        gpuowl.cpp
        k.cpp
        kernel.h
        plan.cpp
        Pm1Plan.cpp
        Pm1Plan.h
        primenet.py
        Primes.cpp
        Primes.h
        PrpGpu.h
        selftest.h
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
        trig.cpp
        Worktodo.cpp
        worktodo.h)

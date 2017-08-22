# gpuOwL
gpuOwL is a Mersenne prime tester for GPUs (OpenCL).
See the GIMPS project for context: http://mersenne.org/

## Selftest
To test the correct function of the software and to validate the GPU, run a selftest:
```
gpuowl -selftest
```

## Select logging step
To obtain more frequent output, set a lower logging step than the default 20000
```
gpuowl -logstep 5000
```

## Usage
* Get exponents for testing from GIPMS Manual Testing ( http://mersenne.org/ ). gpuOwL best handles exponents 70M - 77M.
* Copy the lines from GIMPS to a file named 'worktodo.txt'
* Run gpuowl. It prints progress report on stdout and in gpuowl.log, and writes result lines to results.txt
* Submit the result lines from results.txt to http://mersenne.org/ manual testing.


## Build
To build simply invoke "make" (or look inside the Makefile for a manual build).

* a C++ compiler (e.g. gcc, clang)
* an OpenCL implementation (which provides the **libOpenCL** library), e.g. AMDGPU-Pro
* lib GMP (GNU MP)

## gpuowl -help outputs:

```
gpuOwL v0.6 GPU Lucas-Lehmer primality checker; Mon Aug 21 23:47:40 2017
Command line options:
-logstep  <N>     : to log every <N> iterations (default 20000)
-savestep <N>     : to persist checkpoint every <N> iterations (default 500*logstep == 10000000)
-checkstep <N>    : do Jacobi-symbol check every <N> iterations (default 50*logstep == 1000000)
-uid user/machine : set UID: string to be prepended to the result line
-cl "<OpenCL compiler options>", e.g. -cl "-save-temps=tmp/ -O2"
-selftest         : perform self tests from 'selftest.txt'
                    Self-test mode does not load/save checkpoints, worktodo.txt or results.txt.
-time kernels     : to benchmark kernels (logstep must be > 1)
-legacy           : use legacy kernels

-device <N>       : select specific device among:
    0 : 64x1630MHz gfx900; OpenCL 1.2
```

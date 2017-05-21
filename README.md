# gpuOwL
gpuOwL is a fast GPU Lucas-Lehmer primality test.

### Prerequisites:
* a C++ compiler (e.g. gcc, clang)
* an OpenCL implementation (which provides the **libOpenCL** library), e.g. AMDGPU-Pro

The build is just a C++ compiler invocation on gpuowl.cpp, specifying the OpenCL library in the -L to the compiler, e.g.:

g++ gpuowl.cpp -ogpuowl -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -lOpenCL

## Usage
* Get exponents for testing from GIPMS Manual Testing ( http://mersenne.org/ ). gpuOwL best handles exponents 70M - 77M.
* Copy the lines from GIMPS to a file named 'worktodo.txt'
* Run gpuowl. It prints progress report on stdout and in gpuowl.log, and writes result lines to results.txt
* Submit the result lines from results.txt to http://mersenne.org/ manual testing.

## gpuowl -help outputs:

'''
gpuOwL v0.2 GPU Lucas-Lehmer primality checker; Sun May 21 20:58:26 2017
Command line options:
-cl "<OpenCL compiler options>"
    All the cl options must be included in the single argument following -cl
    e.g. -cl "-D LOW_LDS -D NO_ERR -save-temps=tmp/ -O2"
        -save-temps or -save-temps=tmp or -save-temps=tmp/ : save ISA
        -D LOW_LDS : use a variant of the amalgamation kernel with lower LDS
        -D NO_ERR  : do not compute maximum rounding error
-logstep  <N> : to log every <N> iterations (default 20000)
-savestep <N> : to persist checkpoint every <N> iterations (default 500*logstep == 10000000)
-time kernels : to benchmark kernels (logstep must be > 1)
-selftest     : perform self tests from 'selftest.txt'
-legacy       : use legacy (old) kernels
                Self-test mode does not load/save checkpoints, worktodo.txt or results.txt.
-device   <N> : select specific device among:
    0 : 64x1000MHz Fiji; OpenCL 1.2 AMD-APP (2348.3)
    1 : 16x2400MHz Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz; OpenCL 1.2 AMD-APP (2348.3)

Files used by gpuOwL:
    - worktodo.txt : contains exponents to test "Test=N", one per line
    - results.txt : contains LL results
    - cN.ll : the most recent checkpoint for exponent <N>; will resume from here
    - tN.ll : the previous checkpoint, to be used if cN.ll is lost or corrupted
    - bN.ll : a temporary checkpoint that is renamed to cN.ll once successfully written
    - sN.iteration.residue.ll : a persistent checkpoint at the given iteration

The lines in worktodo.txt must be of one of these forms:
Test=70100200
Test=3181F68030F6BF3DCD32B77337D5EF6B,70100200,75,1
DoubleCheck=3181F68030F6BF3DCD32B77337D5EF6B,70100200,75,1
Test=0,70100200,0,0
'''

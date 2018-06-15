# gpuOwl
gpuOwl is a Mersenne (see http://mersenne.org/ ) primality tester implemented in OpenCL, that works well on AMD GPUs.

gpuOwl implements the PRP test with a powerful self-validating algorithm that protects agains errors.
gpuOwl uses FFT transforms of size 8M and 16M, and is best used with Mersenne exponents in the vicinity of 150M and 300M.

## Files used by gpuOwl
* worktodo.txt : contains exponents to test, one entry per line
* results.txt : contains the results
* N.owl : the most recent checkpoint for exponent <N>; will resume from here
* N-prev.owl : the previous checkpoint, to be used if N.ll is lost or corrupted
* N.iteration.owl : a persistent checkpoint at the given iteration

## worktodo.txt
The lines in worktodo.txt must be of one of these forms:
* 70100200
* PRP=FCECE568118E4626AB85ED36A9CC8D4F,1,2,77936867,-1,75,0

The first form indicates just the exponent to test, while the form starting with PRP indicates both the
exponent and the assignment ID (AID) from PrimeNet.

## Usage
* Make sure that the gpuowl.cl file is in the same folder as the executable
* Get "PRP smallest available first time tests" assignments from GIMPS Manual Testing ( http://mersenne.org/ ).
* Copy the assignment lines from GIMPS to a file named 'worktodo.txt'
* Run gpuowl. It prints progress report on stdout and in gpuowl.log, and writes result lines to results.txt
* Submit the result lines from results.txt to http://mersenne.org/ manual testing.


## Build
To build simply invoke "make" (or look inside the Makefile for a manual build).

* a C++ compiler (e.g. gcc, clang)
* an OpenCL implementation (which provides the **libOpenCL** library). Recommended: an AMD GPU with ROCm 1.7.


## See \"gpuowl -h\" for the command line options:

```
gpuOwL v2.0 GPU Mersenne primality checker
Command line options:

-user <name>  : specify the user name.
-cpu  <name>  : specify the hardware name.
-longCarry    : use not-fused carry kernels (may be slower).
-longTail     : use not-fused tail kernels  (may be slower).
-dump <path>  : dump compiled ISA to the folder <path> that must exist.
-verbosity <level> : change amount of information logged. [0-2, default 0].
-device <N>   : select specific device among:
    0 : Vega [Radeon RX Vega] 64 @83:0.0, gfx900 1630MHz
```

## Self-test
Right now there is no explicit self-testing in GpuOwl. Simply start GpuOwl with any valid exponent, and the built-in error
checking kicks in, implicitly validating the computation. If you start seeing output lines with "OK", than it's working correctly.
"EE" lines indicate computation errors.

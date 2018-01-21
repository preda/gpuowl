# GpuOwL
GpuOwL is a Mersenne prime tester for GPUs. See the GIMPS project for context: http://mersenne.org/

GpuOwl is implemented in OpenCL. It is most tested on AMD GPUs.


## PRP: PRobable Prime test
GpuOwl implements a base-3 PRP (probable prime) test. The reason for choosing PRP vs. LL (Lucas Lehmer) is the
availability of a great error checking algoritm for the PRP, which enables very reliable computation on GPUs
regardless of common hardware problems present on GPUs during long computation.


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

GpuOwl also accepts the LL-test format lines from PrimeNet, but support for these may be removed in the future:
* Test=3181F68030F6BF3DCD32B77337D5EF6B,70100200,75,1
* DoubleCheck=3181F68030F6BF3DCD32B77337D5EF6B,70100200,75,1


## Usage
* Make sure that the gpuowl.cl file is in the same folder as the executable
* Get "PRP smallest available first time tests" assignments from GIPMS Manual Testing ( http://mersenne.org/ ). GpuOwL best handles exponents 70M - 78M.
* Copy the assignment lines from GIMPS to a file named 'worktodo.txt'
* Run gpuowl. It prints progress report on stdout and in gpuowl.log, and writes result lines to results.txt
* Submit the result lines from results.txt to http://mersenne.org/ manual testing.


## Build
To build simply invoke "make" (or look inside the Makefile for a manual build).

* a C++ compiler (e.g. gcc, clang)
* an OpenCL implementation (which provides the **libOpenCL** library). Recommended: an AMD GPU with ROCm 1.7.


## See \"gpuowl -h\" for up-to-date information on the command line options:

```
gpuOwL v1.9-e3055e1-mod GPU Mersenne primality checker
Command line options:

-size 2M|4M|8M : override FFT size.
-fft DP|SP|M61|M31  : choose FFT variant [default DP]:
                DP  : double precision floating point.
                SP  : single precision floating point.
                M61 : Fast Galois Transform (FGT) modulo M(61).
                M31 : FGT modulo M(31).
-user <name>  : specify the user name.
-cpu  <name>  : specify the hardware name.
-legacy       : use legacy kernels
-dump <path>  : dump compiled ISA to the folder <path> that must exist.
-verbosity <level> : change amount of information logged. [0-2, default 0].
-device <N>   : select specific device among:
    0 : Vega [Radeon RX Vega] 64 @83:0.0, gfx900 1630MHz
```


## FFT size
GpuOwl internally does repeated multiplication of very large numbers (tens of millions of bits large). The multiplication
is done through a convolution, wich is done through using a pair of FFT and inverse-FFT transforms. Right now GpuOwl only
supports FFT transforms of power-of-two sizes 2M (i.e. 2M == 2^21), 4M, 8M. The size of the FFT is linked to exponent magnitude.
For exponents up to ~78M (but not too small), an FFT size of 4M is adequate. For larger exponents a larger FFT is needed (8M).


## Legacy kernels
GpuOwl allows selection of a different set of kernels, called the "legacy kernels", then the default. These are likely to be a bit
slower, but use a bit less GPU resources.


## Self-test
Right now there is no explicit self-testing in GpuOwl. Simply start GpuOwl with any valid exponent, and the built-in error
checking kicks in, implicitly validating the computation. If you start seeing output lines with "OK", than it's working correctly.
"EE" lines OTOH indicate computation errors, which are automatically retried.

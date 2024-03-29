[![Actions Status](https://github.com/preda/gpuowl/workflows/CI/badge.svg?branch=master)](https://github.com/preda/gpuowl/actions)

# GpuOwl

GpuOwl is a Mersenne primality tester for AMD, Nvidia and Intel GPUs supporting OpenCL.

If you are making source code changes to GpuOwl, please read the [code style](codestyle.md).

## Mersenne primes

Mersenne numbers are numbers of the form $2^p-1$. Some of these are prime numbers, called _Mersenne primes_.

The largest known Mersenne primes are huge numbers. They are extremely difficult to find, and discovering a new Mersenne prime
is a noteworthy achievement. A long-standing distributed computing project named the Great Internet Mersenne Prime Search (GIMPS)
has been searching for Mersenne primes for the last 30 years.

While traditionally the algorithms involved were implemented targeting CPUs, the GPUs have seen increased usage in computing recently
because of their impressive power and wide memory bandwidth, which are advantages relative to CPUs.

GpuOwl is an implementation of some of the algorithms involved in searching for Mersenne primes in the OpenCL language for execution
on modern AMD, NVIDIA and Intel GPUs. GpuOwl runs best on top of the ROCm OpenCL stack.

## Mersenne primality tests

These are the main test involved in Mersenne prime search:

- TF, Trial Factoring
- P-1, Pollard's p-1 factoring
- LL, Lucas-Lehmer primality test
- PRP, probable prime test

### Trial Factoring (TF)

In this test, prime factors of increasingly larger magnitude are tried, checking if they divide the Mersenne candidate M(p).
TF is good as a first line of attack, representing a cheap filter that removes some Mersenne candidates by finding a factor (thus
deciding that the M(p) is not prime). The limitation of TF is that the checking effort grows exponentially with the size of the
factors that are trialed, thus TF remains just a “first line of attack” approach.

### Pollard's P-1 factoring (P-1)

This is a very ingenious, beautiful algorithm for finding factors of Mersenne candidates. It detects a special class of factors
F where F-1 is highly composite (has many factors). P-1 is used as a preliminary filter (much like TF), that removes some Mersenne
candidates, proving them composite by finding a factor.

### Lucas-Lehmer (LL)

This is a test that proves whether a Mersenne number is prime or not, but without providing a factor in the case where it is not prime.
The Lucas-Lehmer test is very simple to describe: iterate the function $f(x)=(x^2 - 2)\ modulo\ M(p)$ starting with the number 4. If
after p-2 iterations the result is 0, then $M(p)$ is certainly prime, otherwise $M(p)$ is certainly not prime.

Lucas-Lehmer, while a very efficient primality test, still takes a rather long time for large Mersenne numbers
(on the order of weeks of intense compute), thus it is only applied to the Mersenne candidates that survived the cheaper preliminary
filters TF and P-1.

### PRP (“the new LL”)

The probable prime test can prove that a candidate is composite (without providing a factor), but does not prove that a candidate
is prime (only stating that it _probably_ is prime) -- although in practice the difference between probable prime and proved
prime is extremely small for large Mersenne candidates.

The PRP test is very similar computationally to LL: PRP iterates $f(x) = x^2\ modulo\ M(p)$ starting from 3. If after p iterations the result is $9\ modulo\ M(p)$, then $M(p)$ is probably prime, otherwise $M(p)$ is certainly not prime. The cost
of PRP is exactly the same as LL.

In practice, PRP is preferred over LL because PRP does have a very strong and useful error-checking technique, which protects effectively against computation errors (which are sometimes common on GPUs).

## GpuOwl: OpenCL GPU Mersenne primality testing

GpuOwl implements the PRP and P-1 tests. It also implemented, at various points in the past, LL and TF but these are not active now
in GpuOwl. For double check (DC) LL tests, see the [v6 branch](https://github.com/preda/gpuowl/tree/v6) (version 6.11-382) and for first time LL tests, see the [LL branch](https://github.com/preda/gpuowl/tree/LL) (version 0.6).

Let us consider the PRP test, to get an idea of what GpuOwl does under the hood.

PRP uses what is called a _modular squaring_, computing $f(x) = x^2\ modulo\ M(p)$, starting from 3 (where x is an integer).

The problem is in the size of the integer x that is to be squared, which is about 100 million bits in size.

How do we compute efficiently the square of a 100 million bits integer? It turns out that one of the fastest multiplication algorithms
for huge numbers consists in doing a convolution, which involves a direct and an inverse FFT transform, with a simple element-wise
multiplication in the FFT domain.

And this is exactly what GpuOwl does: it implements, as building blocks, efficient huge FFT transforms. Many algorithmic tricks
are also used to speed up computation, e.g. the “Irrational Base Discrete Weighted Transform” (IBDWT) described by Richard Crandall.

## Files used by GpuOwl

- `worktodo.txt` : contains exponents to test, one entry per line
- `results.txt` : contains the results
- `N.owl` : the most recent checkpoint for exponent $N$; will resume from here
- `N-prev.owl` : the previous checkpoint, to be used if N.ll is lost or corrupted
- `N.iteration.owl` : a persistent checkpoint at the given iteration
- `config.txt`: a configuration file for persistent command-line arguments

## `worktodo.txt`

The lines in `worktodo.txt` must be of one of these forms:

- `70100200`
- `PRP=1,2,77936867,-1,75,0`
- `PRP=N/A,1,2,77936867,-1,75,0`
- `PRP=FCECE568118E4626AB85ED36A9CC8D4F,1,2,77936867,-1,75,0`

The first form indicates just the exponent to test, while the form starting with `PRP=` indicates the exponent and optionally the assignment ID (AID) from PrimeNet. The `PRPDC=` prefix can be used instead for PRP DC assignments.

## Usage

- Get "PRP smallest available first time tests" assignments from [GIMPS Manual Testing](https://www.mersenne.org/manual_assignment/).
- Copy the assignment lines from GIMPS to a file named `worktodo.txt`.
- Run `gpuowl`. It prints progress report on stdout and in `gpuowl.log`, and writes result lines to `results.txt`.
- Submit the result lines from `results.txt` to [GIMPS Manual Results](https://www.mersenne.org/manual_result/).

## Build

Prerequisites (please install these):

- the GNU Multiple Precision (GMP) 6.1 library `libgmp-dev`
- a C++20 compiler (e.g. GCC, Clang)
- an OpenCL implementation (which provides the **libOpenCL** library). Recommended: an AMD GPU with ROCm 1.7.

### Meson build

Example build steps on linux:

```bash
cd gpuowl
mkdir build
cd build
meson ..
ninja
```

What the previous commands do:

- go to gpuowl source directory
- create a subdirectory named "build"
- go to the build directory
- invoke meson, passing as argument the gpuowl source directory (.. in this situation)
- run ninja to build

### Make build

To build simply invoke "`make`" (or look inside the Makefile for a manual build).

> See \"`gpuowl -h`\" for the command line options.

## Self-test

Simply start GpuOwl with any valid exponent, and the built-in error checking kicks in, validating the computation. If you start seeing output lines with "OK", than it is working correctly. "EE" lines indicate computation errors.

## Command-line Arguments

```ascii
GpuOwl is an OpenCL program for primality testing of Mersenne numbers (numbers of the form 2^n - 1).
To run GpuOwl you need a computer with one or more discrete GPUs.

To check that OpenCL is installed correctly use the command "clinfo". If clinfo does not find any
devices or otherwise fails, GpuOwl will not run -- you need to first fix the OpenCL installation
to get clinfo to find devices.

GpuOwl runs best on Linux with the ROCm OpenCL stack, but can also run on Windows and on Nvidia GPUs.

For more information about Mersenne primes search see https://www.mersenne.org/

First step: run "gpuowl -h". If this displays a list of OpenCL devices at the end, it means that
gpuowl is detecting the GPUs correctly and will be able to run.

To use GpuOwl you need to create a file named "worktodo.txt" containing the exponent to be tested.
The tool primenet.py (found at gpuowl/tools/primenet.py) can be used to automatically obtain tasks
from the mersenne project and add them to worktodo.txt.

The configuration options listed below can be passed on the command line or can be put in a file
named "config.txt" in the gpuowl run directory.


-dir <folder>      : specify local work directory (containing worktodo.txt, results.txt, config.txt, gpuowl.log)
-pool <dir>        : specify a directory with the shared (pooled) worktodo.txt and results.txt
                     Multiple GpuOwl instances, each in its own directory, can share a pool of assignments and report
                     the results back to the common pool.
-user <name>       : specify the user name.
-cpu  <name>       : specify the hardware name.
-time              : display kernel profiling information.
-fft <spec>        : specify FFT e.g.: 1152K, 5M, 5.5M, 256:10:1K
-block <value>     : PRP error-check block size. Must divide 10'000.
-log <step>        : log every <step> iterations. Multiple of 10'000.
-carry long|short  : force carry type. Short carry may be faster, but requires high bits/word.
-B1                : P-1 B1 bound
-B2                : P-1 B2 bound
-rB2               : ratio of B2 to B1. Default 20, used only if B2 is not explicitly set
-prp <exponent>    : run a single PRP test and exit, ignoring worktodo.txt
-verify <file>     : verify PRP-proof contained in <file>
-proof <power>     : By default a proof of power 8 is generated, using 3GB of temporary disk space for a 100M exponent.
                     A lower power reduces disk space requirements but increases the verification cost.
                     A proof of power 9 uses 6GB of disk space for a 100M exponent and enables faster verification.
-autoverify <power> : Self-verify proofs generated with at least this power. Default 9.
-tmpDir <dir>      : specify a folder with plenty of disk space where temporary proof checkpoints will be stored, default '.'.
-mprimeDir <dir>   : folder where an instance of Prime95/mprime can be found (for P-1 second-stage)
-results <file>    : name of results file, default 'results.txt'
-iters <N>         : run next PRP test for <N> iterations and exit. Multiple of 10000.
-maxAlloc <size>   : limit GPU memory usage to size, which is a value with suffix M for MB and G for GB.
                     e.g. -maxAlloc 2048M or -maxAlloc 3.5G
-save <N>          : specify the number of savefiles to keep (default 20).
-noclean           : do not delete data after the test is complete.
-from <iteration>  : start at the given iteration instead of the most recent saved iteration
-yield             : enable work-around for Nvidia GPUs busy wait. Do not use on AMD GPUs!

-use <define>      : comma separated list of defines for configuring gpuowl.cl, such as:
  -use FAST_BARRIER: on AMD Radeon VII and older AMD GPUs, use a faster barrier(). Do not use
                     this option on Nvidia GPUs or on RDNA AMD GPUs where it produces errors
                     (which are nevertheless detected).
  -use NO_ASM      : do not use __asm() blocks (inline assembly)
  -use CARRY32     : force 32-bit carry (-use STATS=21 offers carry range statistics)
  -use CARRY64     : force 64-bit carry (a bit slower but no danger of carry overflow)
  -use TRIG_COMPUTE=0|1|2 : select sin/cos tradeoffs (compute vs. precomputed)
                     0 uses precomputed tables (more VRAM access, less DP compute)
                     2 uses more DP compute and less VRAM table access
  -use DEBUG       : enable asserts in OpenCL kernels (slow)
  -use STATS       : enable roundoff (ROE) or carry statistics logging.
                     Allows selecting among the the kernels CarryFused, CarryFusedMul, CarryA, CarryMul using the bit masks:
                     1 = CarryFused
                     2 = CarryFusedMul
                     4 = CarryA
                     8 = CarryMul
                    16 = analyze Carry instead of ROE
                     (the bit mask 16 selects Carry statistics, otherwise ROE statistics)
                     E.g. STATS=15 enables ROE stats for all the four kernels above.
                          STATs=21 enables Carry stats for the CarryFused and CarryA.
                     For carry, the range [0, 2^32] is mapped to [0.0, 1.0] float values; as such the max carry
                     that fits on 32bits (i.e. 31bits absolute value) is mapped to 0.5

-unsafeMath        : use OpenCL -cl-unsafe-math-optimizations (use at your own risk)

-device <N>        : select the GPU at position N in the list of devices
-uid    <UID>      : select the GPU with the given UID (on ROCm/AMDGPU, Linux)
-pci    <BDF>      : select the GPU with the given PCI BDF, e.g. "0c:00.0"
```

Device numbers start at zero.

## `Primenet.py` Arguments

```ascii
-h, --help            show this help message and exit
-u USERNAME           Primenet user name
-p PASSWORD           Primenet password
-t TIMEOUT            Seconds to sleep between updates
--dirs DIR [DIR ...]  GpuOwl directories to scan
--tasks NTASKS        Number of tasks to fetch ahead
-w {PRP,PM1,LL_DC,PRP_DC,PRP_WORLD_RECORD,PRP_100M}   GIMPS work type
```

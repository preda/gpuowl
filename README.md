# GpuOwl

GpuOwl is a Mersenne primality tester for AMD GPUs.

## Mersenne primes
Mersenne numbers are numbers of the form 2^p -1. Some of these are prime numbers, called <em>Mersenne primes</em>.

The largest known Mersenne primes are huge numbers. They are extremely difficult to find, and discovering a new Mersenne prime
is a noteworthy achievement. A long-standing distributed compting project named the Great Internet Mersenne Prime Search (GIMPS)
has been searching for Mersenne primes for the last 30 years.

While traditionally the algorithms involved were implemented targeting CPUs, the GPUs have seen increased usage in computing recently
because of their impressive power and wide memory bandwidth, which are advantages relative to CPUs.

GpuOwl is an implementation of some of the algorithms involved in searching for Mersenne primes in the OpenCL language for execution
on modern AMD GPUs. GpuOwl runs best on top of the ROCm OpenCL stack.

## Mersenne primality tests
These are the main test involved in Mersenne prime search:
* TF, Trial Factoring
* P-1, Pollard's p-1 factoring
* LL, Lucas-Lehmer primality test
* PRP, probable prime test

### Trial Factoring (TF)
In this test prime factors of increasingly larger magnitude are tried, checking if they divide the Mersenne candidate M(p).
TF is good as a first line of attack, representing a cheap filter that removes some Mersenne candidates by finding a factor (thus
deciding that the M(p) is not prime). The limitation of TF is that the checking effort grows exponentially with the size of the
factors that are trialed, thus TF remains just a "first line of attack" approach.

### Pollard's P-1 factoring (P-1)
This is a very ingenious, beautiful algorithm for finding factors of Mersenne candidates. It detects a special class of factors
F where F-1 is higly composite (has many factors). P-1 is used as a preliminary filter (much like TF), that removes some Mersenne
candidates, proving them composite by finding a factor.

### Lucas-Lehmer (LL)
This is a test that proves whether a Mersenne number is prime or not, but without providing a factor in the case where it's not prime.
The Lucas-Lehmer test is very simple to describe: iterate the function f(x)=(x^2 - 2) modulo M(p) starting with the number 4. If
after p-1 iterations the result is 0, then M(p) is certainly prime, otherwise M(p) is certainly not prime.

Lucas-Lehmer, while a very efficient primality test, still takes a rather long time for large Mersenne numbers
(on the order of weeks of intense compute), thus it is only applied to the Mersenne candidates that survived the cheaper preliminary
filters TF and P-1.

### PRP ("the new LL")
The probable prime test can prove that a candidate is composite (without providing a factor), but does not prove that a candidate
is prime (only stating that it <em>probably</em> is prime) -- although in practice the difference between probable prime and proved
prime is extremely small for large mersenne candidates.

The PRP test is very similar computationally to LL: PRP iterates f(x) = x^2 modulo M(p) starting from 3, for p iterations. The cost
of PRP is exacly the same as LL.

In practice PRP is preferred over LL because PRP does have a very strong and useful error-checking technique, which protects effectivelly against computation errors (which are sometimes common on GPUs).

## GpuOwl: OpenCL GPU Mersenne primality testing
GpuOwl implements the PRP and P-1 tests. It also implemented, at various points in the past, LL and TF but these are not active now
in GpuOwl.

Let's consider the PRP test, to get an idea of what GpuOwl does under the hood.

PRP uses what is called a <em>modular squaring</em>, computing f(x) = x^2 modulo M(p), starting from 3 (where x is an integer).

The problem is in the size of the integer x that is to be squared, which is on the order of 100 million bits in size.

How do we compute efficiently the square of a 100 million bits integer? It turns out that one of the fastest multiplication algorithms
for huge numbers consists in doing a convolution, which involves a direct and an inverse FFT transform, with a simple element-wise
multiplication in the FFT domain.

And this is exacly what GpuOwl does: it implements, as building blocks, efficient huge FFT transforms. Many algorithmic tricks
are also used to speed up computation, e.g. the "Irrational Base Discrete Weighted Transform" (IBDWT) described by Richard Crandall.



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

* the library libgmp-dev
* a C++ compiler (e.g. gcc, clang)
* an OpenCL implementation (which provides the **libOpenCL** library). Recommended: an AMD GPU with ROCm 1.7.

## See \"gpuowl -h\" for the command line options.

## Self-test
Simply start GpuOwl with any valid exponent, and the built-in error checking kicks in, validating the computation. If you start seeing output lines with "OK", than it's working correctly. "EE" lines indicate computation errors.

## Command-line Arguments
-user \<name\>       : specify the user name.\
-cpu  \<name\>       : specify the hardware name.\
-time              : display kernel profiling information.\
-fft \<size\>        : specify FFT size, such as: 5000K, 4M, +2, -1.\
-block \<value\>     : PRP GEC block size. Default 400. Smaller block is slower but detects errors sooner.\
-carry long|short  : force carry type. Short carry may be faster, but requires high bits/word.\
-B1                : P-1 B1, default 500000\
-rB2               : ratio of B2 to B1, default 30\
-device \<N\>        : select a specific device:\
  \
Device numbers start at zero.


[![Actions Status](https://github.com/preda/gpuowl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/preda/gpuowl/actions/workflows/ci.yml)

## Must read papers

### Multiplication by FFT

- [Discrete Weighted Transforms and Large Integer Arithmetic](https://www.ams.org/journals/mcom/1994-62-205/S0025-5718-1994-1185244-1/S0025-5718-1994-1185244-1.pdf), Richard Crandall and Barry Fagin, 1994
- [Rapid Multiplication Modulo the Sum And Difference of Highly Composite Numbers](https://www.daemonology.net/papers/fft.pdf), Colin Percival, 2002

### P-1

- [An FFT Extension to the P-1 Factoring Algorithm](https://www.ams.org/journals/mcom/1990-54-190/S0025-5718-1990-1011444-3/S0025-5718-1990-1011444-3.pdf), Montgomerry & Silverman, 1990
- [Improved Stage 2 to P+/-1 Factoring Algorithms](https://inria.hal.science/inria-00188192v3/document), Montgomerry & Kruppa, 2008


# PRPLL

## PRobable Prime and Lucas-Lehmer mersenne categorizer
(pronounced *purrple categorizer*)

PRPLL implements two primality tests for Mersenne numbers: PRP ("PRobable Prime") and LL ("Lucas-Lehmer") as the name suggests.

PRPLL is an OpenCL (GPU) program for primality testing Mersenne numbers.

## Build

Invoke `make` in the source directory.
Multi-threaded build (`make -j$(nproc)`) is supported for the impatient.

## Pre-built binaries

You can get pre-built binaries from [nightly.link](https://nightly.link/preda/gpuowl/workflows/ci/master). This is identical to going to the "Actions" tab on GitHub, choosing the latest commit, then to section "Artifacts".

## Use

See `prpll -h` for the command line options.

You may want to copy the `tune.txt` found in project root directory alongside the `prpll` binary.
Alternatively you may want to run `prpll -tune` to generate your own.

`prpll` may have additional dependencies. If you don't have them, the program will not launch.
On Linux and MSYS2 (Windows), you can use `ldd` to list what libraries it requires and that you may be missing.
On macOS `otool -L` can be used.

## Work types supported

For Mersenne primes search, the PRP test is by far preferred over LL, such that LL is not used anymore for search.
This is because there is a simple way to verify the results of a PRP computation, unlike LL which is verified by full recomputation.
But PRP does not definitely confirm a number as prime, so LL is still used to verify a probable prime found by PRP (which is a very
rare occurence).

PRPLL supports LL, PRP, and CERT jobs.

### Lucas-Lehmer (LL)
This is a test that proves whether a Mersenne number is prime or not, but without providing a factor in the case where it is not prime.
The Lucas-Lehmer test is very simple to describe: iterate the function f(x)=(x^2 - 2) modulo M(p) starting with the number 4. If
after p-2 iterations the result is 0, then M(p) is certainly prime, otherwise M(p) is certainly not prime.

Lucas-Lehmer, while a very efficient primality test, still takes a rather long time for large Mersenne numbers
(on the order of weeks of intense compute), thus it is only applied to the Mersenne candidates that survived the cheaper preliminary
filters TF and P-1.

### PRP
The probable prime test can prove that a candidate is composite (without providing a factor), but does not prove that a candidate
is prime (only stating that it _probably_ is prime) -- although in practice the difference between probable prime and proved
prime is extremely small for large Mersenne candidates.

The PRP test is very similar computationally to LL: PRP iterates f(x) = x^2 modulo M(p) starting from 3. If after p iterations the result is 9 modulo M(p), then M(p) is probably prime, otherwise M(p) is certainly not prime. The cost
of PRP is exactly the same as LL.

The PRP test also generates a proof file based on the Pietrzak VDF scheme authenticating the calculation leading to the reported
residue. This file is meant to be uploaded to PrimeNet alongside the JSON result output.

### CERT
PrimeNet hands out work containing other people's cert files to trusted computers. These computers verify the Pietrzak proof of the
modular exponentiation process. PRPLL is able to process CERT work since October 2024 (version TO_BE_RELEASED).

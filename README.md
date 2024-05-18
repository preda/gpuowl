[![Actions Status](https://github.com/preda/gpuowl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/preda/gpuowl/actions/workflows/ci.yml)

# PRPLL
## PRobable Prime and Lucas-Lehmer mersenne categorizer
(pronounced *purrple categorizer*)

PRPLL implements two primality tests for Mersenne numbers: PRP ("PRobable Prime") and LL ("Lucas-Lehmer") as the name suggests.

PRPLL is an OpenCL (GPU) program for primality testing Mersenne numbers.


## Build

Invoke `make` in the source directory.


## Use
See `prpll -h` for the command line options.


## Why LL

For Mersenne primes search, the PRP test is by far preferred over LL, such that LL is not used anymore for search.
But LL is still used to verify a prime found by PRP (which is a very rare occurence).


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

// Copyright (C) Mihai Preda

#include "Primes.h"

#include <cassert>

Primes::Primes() {
  sieve.set();

  for (u32 i = 0; i < sieve.size(); ++i) {
    if (sieve[i]) {
      u32 n = 2 * i + 3;
      for (u32 k = i + n; k < sieve.size(); k += n) { sieve.reset(k); }
    }
  }
}

bool Primes::isPrimeOdd(u32 n) const {
  assert(n % 2); // must be odd to call here

  if (n < 3) { return false; }
  for (u32 k = 0; k < sieve.size(); ++k) {
    if (sieve[k]) {
      u32 p = k * 2 + 3;
      if (p * p > n) { return true; }
      if (n % p == 0) { return false; }
    }
  }
  assert(false);
  return false;
}

bool Primes::isPrime(u32 n) const {
  return (n%2 && isPrimeOdd(n)) || (n == 2);
}

u32 Primes::prevPrime(u32 n) const {
  --n;
  if (n % 2 == 0) { --n; }

  for (; n >= 2; n -= 2) { if (isPrimeOdd(n)) { return n; } }
  assert(false);
  return 0;
}

u32 Primes::nextPrime(u32 n) const {
  ++n;
  if (n % 2 == 0) { ++n; }
  for (; ; n += 2) { if (isPrimeOdd(n)) { return n; }}
  assert(false);
  return 0;
}

u32 Primes::nearestPrime(u32 n) const {
  if (isPrime(n)) { return n; }
  u32 a = prevPrime(n);
  u32 b = nextPrime(n);
  assert(a < n && n < b);
  return n-a < b-n ? a : b;
}

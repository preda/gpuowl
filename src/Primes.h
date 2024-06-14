// Copyright (C) Mihai Preda

#pragma once

#include <bitset>
#include "common.h"

class Primes {
  std::bitset<50000> sieve;
  bool isPrimeOdd(u32 n) const;

public:
  Primes();

  bool isPrime(u32 n) const;
  u32 prevPrime(u32 n) const;
  u32 nextPrime(u32 n) const;
  u32 nearestPrime(u32 n) const;
};

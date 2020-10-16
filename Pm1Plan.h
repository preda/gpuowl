// Copyright Mihai Preda

#pragma once

#include "common.h"

#include <vector>
#include <bitset>

struct PlanStats {
  u32 nPrimes, firstPrime, lastPrime;
  float cost;
  vector<u32> nPairs;
  u32 nSingle;
  u32 nBlocks;
};

class Pm1Plan {
  static constexpr const u32 MAX_BUFS = 1024;

public:
  using BitBlock = bitset<MAX_BUFS>;
  
  const u32 nBuf;  // number of precomputed "big" GPU buffers

  vector<bool> primeBits; // Bits indicating primes in [B1,B2].
  
  vector<u32> makeJset();  // A set of nBuf values that are relative prime with "D".
  
  // vector<bool> makePrimeBits();  // Generate a vector of bits indicating primes between B1 and B2.
  
  u32 primeAfter(u32 b) const;   // return smallest prime > b
  u32 primeBefore(u32 b) const;  // return largest prime < b

  // The largest block that covers "b" (the initial block).
  u32 lowerBlock(u32 b) const;
  
  // The smallest block that cover "b" (the final block).
  u32 upperBlock(u32 b) const;

  // Repeatedly divide "pos" by the smallest prime not-factor of D.
  u32 reduce(u32 pos) const;

  // Repeatedly divide "pos" by "F".
  template<u32 F> u32 reduce(u32 pos) const;

  // Returns the prime hit by "a", or 0.
  u32 hit(const vector<bool>& primes, u32 a);

  template<typename Fun>
  void scan(const vector<bool>& primes, u32 beginBlock, vector<Pm1Plan::BitBlock>& selected, Fun fun);
  
public:
  static u32 minBufsFor(u32 D);
  static u32 getD(u32 argsD, u32 nBufs) { return argsD ? argsD : (nBufs >= minBufsFor(330) ? 330 : 210); }

  // Simple Erathostene's sieve restricted to the range [B1, B2].
  // Returns a vector of bits.
  // Only the bits between B1 and B2 are set where there is a prime.
  static vector<bool> sieve(u32 B1, u32 B2);

  const u32 D;
  const u32 B1;
  const u32 B2;
  const vector<u32> jset; // The set of relative primes to "D" corresponding to the precomputed buffers.

  
  Pm1Plan(u32 D, u32 nBuf, u32 B1, u32 B2);
  Pm1Plan(u32 D, u32 nBuf, u32 B1, u32 B2, vector<bool>&& primeBits);

  // Returns a sequence of BitBlocks, one entry per block starting with block=0.
  // Each BitBlock has a bit set if the corresponding buffer is selected for multiplication.
  // Also return beginBlock.
  pair<u32, vector<BitBlock>> makePlan(PlanStats* stats = nullptr);
};

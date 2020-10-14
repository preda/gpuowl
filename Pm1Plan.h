// Copyright Mihai Preda

#pragma once

#include "common.h"

#include <vector>
#include <bitset>

class Args;

class Pm1Plan {
  static constexpr const u32 MAX_BUFS = 1024;
  
  const u32 nBuf;  // number of precomputed "big" GPU buffers

  vector<u32> makeJset();  // A set of nBuf values that are relative prime with "D".
  
  vector<bool> makePrimeBits();  // Generate a vector of bits indicating primes between B1 and B2.
  
  u32 primeAfter(u32 b) const;   // return smallest prime > b
  u32 primeBefore(u32 b) const;  // return largest prime < b

  // The largest block that covers "b" (the initial block).
  u32 lowerBlock(u32 b) const { return (b + jset.back()) / D; }
  
  // The smallest block that cover "b" (the final block).
  u32 upperBlock(u32 b) const { return (b - jset.back()) / D + 1; }

  // Repeatedly divide "pos" by the smallest prime not-factor of D.
  u32 reduce(u32 pos) const;

  // Repeatedly divide "pos" by "F".
  template<u32 F> u32 reduce(u32 pos) const;
  
public:
  using BitBlock = bitset<MAX_BUFS>;

  const u32 D;
  const u32 B1;
  const u32 B2;
  const vector<u32> jset; // The set of relative primes to "D" corresponding to the precomputed buffers.
  const vector<bool> primeBits; // Bits indicating primes in [B1,B2].
  
  Pm1Plan(const Args& args, u32 nBuf, u32 B1, u32 B2);

  // Returns a sequence of BitBlocks, one entry per block starting with block=0.
  // Each BitBlock has a bit set if the corresponding buffer is selected for multiplication.
  vector<BitBlock> makePlan();

  // The first block to use, given that all primes <= doneB2 have been tested.
  u32 getStartBlock(u32 doneB2) const;
};

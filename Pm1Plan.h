// Copyright Mihai Preda
#pragma once

#include "common.h"
#include "AllocTrac.h"

#include <vector>
#include <tuple>
#include <array>
#include <cassert>
#include <numeric>

template<u32 D> constexpr bool isRelPrime(u32 j);
template<> constexpr bool isRelPrime<210>(u32 j) { return j % 2 && j % 3 && j % 5 && j % 7; }
template<> constexpr bool isRelPrime<2310>(u32 j) { return isRelPrime<210>(j) && j % 11; }
template<> constexpr bool isRelPrime<30030>(u32 j) { return isRelPrime<2310>(j) && j % 13; }

bool isRelPrime(u32 D, u32 j) {
  return (D == 210) ? isRelPrime<210>(j) : ((D == 2310) ? isRelPrime<2310>(j) : isRelPrime<30030>(j));
}

// Simple Erathostene's sieve.
vector<bool> primeBits(u32 B1, u32 B2) {
  assert(B1 < B2);

  vector<bool> bits(B2 + 1);
  bits[0] = bits[1] = true;
  for (u32 p = 0; p <= B2; ++p) {
    while (p <= B2 && bits[p]) { ++p; }

    if (p > B2) { break; }

    if (p <= B1) { bits[p] = true; }

    if (u64(p) * p <= B2) { for (u32 i = p * p; i <= B2; i += p) { bits[i] = true; }}
  }
  bits.flip();
  return bits;
}

class Pm1Plan {
  template<typename T>
  static u32 sum(const T &v) { return accumulate(v.begin(), v.end(), 0); }

public:
  const u32 D;
  vector<u32> jset;

  using BitBlock = vector<bool>;
    
  Pm1Plan(u32 nBuf) : D{nBuf >= 2880 ? 30030u : (nBuf >= 240 ? 2310u : 210u)} {
    assert(nBuf >= 24);
    log("using %u buffers\n", nBuf);
    for (u32 j = 1; jset.size() < nBuf; j += 2) {
      if (isRelPrime(D, j)) { jset.push_back(j); }
    }
  }

  u32 lowerBlock(u32 b) { return (b + jset.back()) / D; }
  u32 upperBlock(u32 b) { return (b - jset.back()) / D + 1; }
  
  tuple<u32, vector<BitBlock>> makePlan(u32 B1, u32 doneB2, u32 B2) {
    if (doneB2 == 0) { doneB2 = B1; }
    assert(doneB2 >= B1 && doneB2 < B2);
    
    u32 beginBlock = lowerBlock(doneB2);
    u32 endBlock   = upperBlock(B2) + 1;
    assert(beginBlock < endBlock);
    
    u32 tweakB1 = beginBlock * D - jset.back();
    u32 tweakB2 = (endBlock - 1) * D + jset.back();
    
    assert(tweakB1 > 0 && tweakB2 > tweakB1);
    
    vector<bool> bits{primeBits(tweakB1, tweakB2)};
    
    u32 nPrimes = sum(bits);

    const u32 n = jset.size();

    vector<BitBlock> selected;
    for (u32 block = beginBlock; block < endBlock; ++block) { selected.push_back(BitBlock(n)); }

    u32 nDouble = 0;
    for (u32 block = beginBlock; block < endBlock; ++block) {
      BitBlock& blockBits = selected[block - beginBlock];
      
      for (u32 pos = 0; pos < n; ++pos) {
        u32 j = jset[pos];
        assert(j >= 1);
        u32 ia = block * D - j;
        u32 ib = block * D + j;
        bool a = bits[ia];
        bool b = bits[ib];
        if (a && b) {
          blockBits[pos] = true;
          ++nDouble;
          bits[ia] = false;
          bits[ib] = false;
        }
      }
    }

    u32 nSingle = 0;
    for (u32 block = beginBlock; block < endBlock; ++block) {
      BitBlock& blockBits = selected[block - beginBlock];
      
      for (u32 pos = 0; pos < n; ++pos) {
        u32 j = jset[pos];
        assert(j >= 1);
        u32 ia = block * D - j;
        u32 ib = block * D + j;
        bool a = bits[ia];
        bool b = bits[ib];
        if (a || b) {
          blockBits[pos] = true;
          ++nSingle;
          bits[ia] = false;
          bits[ib] = false;
        }
      }
    }

    assert(sum(bits) == 0); // all primes covered.
  
    u32 total = nSingle + nDouble;
    log("B1=%u, B2=%u, D=%u: %u primes in [%u, %u], selected %u (%.1f%%) (%u doubles + %u singles)\n",
        B1, B2, D, nPrimes, tweakB1, tweakB2, total, total / float(nPrimes) * 100, nDouble, nSingle);
    return {beginBlock, selected};
  }
};

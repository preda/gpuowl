// Copyright Mihai Preda
#pragma once

#include "common.h"

#include <vector>
#include <tuple>
#include <bitset>
#include <array>
#include <cassert>
#include <numeric>

template<u32 D> constexpr bool isRelPrime(u32 j);
template<> constexpr bool isRelPrime<210>(u32 j) { return j % 2 && j % 3 && j % 5 && j % 7; }
template<> constexpr bool isRelPrime<2310>(u32 j) { return isRelPrime<210>(j) && j % 11; }

template<u32 D> constexpr u32 getJ();

template<> constexpr u32 getJ<210>() { return 1 * 2 * 4 * 6 / 2; }
template<> constexpr u32 getJ<2310>() { return getJ<210>() * 10; }
static_assert(getJ<2310>() == 240);

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

template<u32 DV>
class Pm1Plan {
  static u32 blockFor(u32 B1) { return (B1 + D / 2) / D; }  

  template<typename T>
  static u32 sum(const T &v) { return accumulate(v.begin(), v.end(), 0); }

public:  
  static constexpr const u32 D = DV;
  static constexpr const u32 J = getJ<D>();
  
  using BitBlock = std::bitset<J>;

  // Generate a P-1 second stage plan.
  // that covers the primes in (B1, B2], with a block of size D.
  // Returns the index of the first block, the total nb. of points selected, and vectors of selected bits per block.
  static std::tuple<u32, vector<BitBlock>> makePm1Plan(u32 B1, u32 doneB2, u32 B2) {
    if (doneB2 == 0) { doneB2 = B1; }
    assert(doneB2 >= B1 && doneB2 < B2);
    
    u32 beginBlock = blockFor(doneB2);
    u32 endBlock   = blockFor(B2 - 1) + 1;
    
    u32 tweakB1 = beginBlock * D - (D / 2);
    u32 tweakB2 = endBlock * D - (D / 2);
    assert(tweakB1 > 0 && tweakB2 > tweakB1);
    
    vector<bool> bits{primeBits(tweakB1, tweakB2)};
    
    u32 nPrimes = sum(bits);

    u32 nSingle = 0, nDouble = 0;
    
    vector<BitBlock> selected;
    auto jset = getJset();
    for (u32 block = beginBlock; block < endBlock; ++block) {
      BitBlock blockBits;
      for (u32 pos = 0; pos < J; ++pos) {
        u32 j = jset[pos];    
        assert(j >= 1 && j < D / 2);
        u32 ia = block * D - j;
        u32 ib = block * D + j;
        bool a = bits[ia];
        bool b = bits[ib];
        if (a || b) {
          blockBits[pos] = true;
          if (a && b) { ++nDouble; } else { ++nSingle; }
          if (a) { bits[ia] = false; }
          if (b) { bits[ib] = false; }
        }
      }
      selected.push_back(std::move(blockBits));
    }

    assert(sum(bits) == 0); // all primes covered.
  
    u32 total = nSingle + nDouble;
    log("B1=%u, B2=%u, D=%u: %u primes in [%u, %u], selected %u (%.1f%%) (%u doubles + %u singles)\n",
        B1, B2, D, nPrimes, tweakB1, tweakB2, total, total / float(nPrimes) * 100, nDouble, nSingle);
    return {beginBlock, selected};
  }

  // JSet : the values 1 <= x < D/2 where GCD(x, D) == 1
  static constexpr array<u32, J> getJset() {
    array<u32, J> jset{};
    u32 pos = 0;
    for (u32 j = 1; j < D / 2; j += 2) { if (isRelPrime<D>(j)) { jset[pos++] = j; }}
    assert(pos == J);
    return jset;
  }
};

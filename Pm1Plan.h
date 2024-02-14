// Copyright Mihai Preda
#pragma once

#include "common.h"

#include <vector>
#include <tuple>
#include <bitset>
#include <array>
#include <cassert>

using std::vector;

// Generate a P-1 second stage plan.
// that covers the primes in (B1, B2], with a block of size D.
// Returns the index of the first block, the total nb. of points selected, and vectors of selected bits per block.
std::tuple<u32, u32, vector<std::bitset<2880>>> makePm1Plan(u32 B1, u32 B2);

constexpr bool isRelPrime(u32 j) { return j % 2 && j % 3 && j % 5 && j % 7 && j % 11 && j % 13; }

// JSet : the values 1 <= x < D/2 where GCD(x, D) == 1
constexpr array<u32, 2880> getJset() {
  u32 D = 30030;
  
  array<u32, 2880> jset{};
  u32 pos = 0;
  for (u32 j = 1; j < D / 2; j += 2) { if (isRelPrime(j)) { jset[pos++] = j; }}
  assert(pos == 2880);
  return jset;
}

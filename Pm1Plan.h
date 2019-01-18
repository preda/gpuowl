// Copyright Mihai Preda
#pragma once

#include "common.h"

#include <vector>

using std::vector;

// A P-1 second stage execution plan.
struct Pm1Plan {
  // The set of values in [1, D/2] that are relative prime to D.
  vector<u32> jset;

  // Blocks of size D starting at blockBegin.
  u32 blockBegin;

  // for each block starting at blockBegin, the indices into jset that were selected.
  vector<vector<u32>> blockJsetPos;
};

// Generate a P-1 second stage plan that covers the primes in (B1, B2], using a block of size D.
Pm1Plan makePm1Plan(u32 D, u32 B1, u32 B2);

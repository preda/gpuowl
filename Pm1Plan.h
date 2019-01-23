// Copyright Mihai Preda
#pragma once

#include "common.h"

#include <vector>

using std::vector;

// Generate a P-1 second stage plan.
// that covers the primes in (B1, B2], with a block of size D.
// Returns the index of the first block, and vectors of selected bits per block.
std::pair<u32, vector<vector<bool>>> makePm1Plan(u32 D, u32 B1, u32 B2);

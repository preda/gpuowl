// Copyright Mihai Preda
#pragma once

#include "common.h"

#include <vector>
#include <tuple>
#include <bitset>

using std::vector;

// Get new bounds close to B1, B2 that offer efficient block layout.
// std::pair<u32, u32> adjustBounds(u32 B1, u32 B2);
u32 adjustBound(u32 B);

// Generate a P-1 second stage plan.
// that covers the primes in (B1, B2], with a block of size D.
// Returns the index of the first block, the total nb. of points selected, and vectors of selected bits per block.
std::tuple<u32, u32, vector<std::bitset<2880>>> makePm1Plan(u32 B1, u32 B2);


vector<u32> getJset();

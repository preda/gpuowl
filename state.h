// Copyright 2017 Mihai Preda.

#pragma once

#include "common.h"
#include <vector>
#include <cmath>
#include <cassert>
#include <cfenv>

vector<u32> compactBits(const vector<int> &dataVect, int E);
vector<int> expandBits(const vector<u32> &compactBits, int N, int E);
u64 residueFromRaw(u32 N, u32 E, const vector<int> &words);

constexpr u32 step(u32 N, u32 E) { return N - (E % N); }

inline u64 FRAC(u32 N, u32 E) {
  // #pragma STDC FENV_ACCESS ON
  std::fesetround(FE_DOWNWARD);
  long double x = ldexpl((E % N) / (long double) N, 64);
  std::fesetround(FE_TONEAREST);
  return x;
}

// Sets the weighting vectors direct A and inverse iA (as per IBDWT).
pair<vector<double>, vector<double>> genWeights(int E, int W, int H);

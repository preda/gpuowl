// Copyright 2017 Mihai Preda.

#pragma once

#include "common.h"
#include <vector>
#include <cmath>
#include <cassert>
#include <cfenv>

vector<u32> compactBits(const vector<int> &dataVect, u32 E);
vector<int> expandBits(const vector<u32> &compactBits, u32 N, u32 E);
u64 residueFromRaw(u32 N, u32 E, const vector<int> &words);

constexpr u32 step(u32 N, u32 E) { return N - (E % N); }
constexpr u32 extra(u32 N, u32 E, u32 k) { return u64(step(N, E)) * k % N; }

inline u64 FRAC(u32 N, u32 E) {
  // #pragma STDC FENV_ACCESS ON
  std::fesetround(FE_DOWNWARD);
  long double x = ldexpl((E % N) / (long double) N, 64);
  std::fesetround(FE_TONEAREST);
  return x;
}

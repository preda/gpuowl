// Copyright 2017 Mihai Preda.

#pragma once

#include "common.h"
#include <vector>
#include <cmath>

vector<u32> compactBits(const vector<int> &dataVect, int E);
vector<int> expandBits(const vector<u32> &compactBits, int N, int E);
u64 residueFromRaw(u32 N, u32 E, const vector<int> &words);

constexpr u32 step(u32 N, u32 E) { return N - (E % N); }
constexpr u64 FRAC(u32 N, u32 E) { return ldexpl((E % N) / (long double) N, 64); }

// Sets the weighting vectors direct A and inverse iA (as per IBDWT).
pair<vector<double>, vector<double>> genWeights(int E, int W, int H);

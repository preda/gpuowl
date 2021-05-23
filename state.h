// Copyright 2017 Mihai Preda.

#pragma once

#include "common.h"
#include <vector>
#include <cmath>
#include <cassert>
#include <cfenv>

vector<u32> compactBits(const vector<i32>& data, const vector<i64>& carries, u32 E);
vector<int> expandBits(const vector<u32> &compactBits, u32 N, u32 E);
u64 residueFromRaw(u32 N, u32 E, const vector<int> &words);

constexpr u32 step(u32 N, u32 E) { return N - (E % N); }
constexpr u32 extra(u32 N, u32 E, u32 k) { return u64(step(N, E)) * k % N; }
constexpr bool isBigWord(u32 N, u32 E, u32 k) { return extra(N, E, k) + step(N, E) < N; }

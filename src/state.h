// Copyright 2017 Mihai Preda.

#pragma once

#include "common.h"
#include <vector>
#include <cmath>
#include <cassert>
#include <cfenv>

vector<u32> compactBits(const vector<int> &dataVect, u32 E);
vector<int> expandBits(const vector<u32> &compactBits, u32 N, u32 E);

constexpr u32 step(u32 N, u32 E) { return N - (E % N); }
constexpr u32 extra(u32 N, u32 E, u32 k) { return u64(step(N, E)) * k % N; }
constexpr bool isBigWord(u32 N, u32 E, u32 k) { return extra(N, E, k) + step(N, E) < N; }
constexpr u32 bitlen(u32 N, u32 E, u32 k) { return E / N + isBigWord(N, E, k); }

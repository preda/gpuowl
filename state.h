// Copyright 2017 Mihai Preda.

#include "common.h"
#include <vector>

#pragma once

vector<u32> compactBits(const vector<int> &dataVect, int E);
vector<int> expandBits(const vector<u32> &compactBits, int N, int E);
u64 residueFromRaw(u32 E, u32 N, const vector<int> &words);
pair<vector<double>, vector<double>> genWeights(int E, int W, int H);

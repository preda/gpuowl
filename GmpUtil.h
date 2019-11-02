// Copyright (C) Mihai Preda.

#include "common.h"

#include <string>
#include <vector>

// return GCD(bits - sub, 2^exp - 1) as a decimal string if GCD!=1, or empty string otherwise.
std::string GCD(u32 exp, const std::vector<u32>& words, u32 sub = 0);

// "Big Endian" means most significant bit at index 0.
vector<bool> powerSmoothBE(u32 exp, u32 B1);

// vector<bool> mulBE(const vector<bool>& a, u64 b) { }

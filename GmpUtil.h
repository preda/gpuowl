// Copyright (C) Mihai Preda.

#pragma once

#include "common.h"
#include <gmpxx.h>

#include <string>
#include <vector>

// return GCD(bits - sub, 2^exp - 1) as a decimal string if GCD!=1, or empty string otherwise.
std::string GCD(u32 exp, const std::vector<u32>& words, u32 sub = 0);

// Represent mpz value as vector of bits with the most significant bit first.
vector<bool> bitsMSB(mpz_class a);

vector<bool> powerSmoothMSB(u32 exp, u32 B1);

// Returns jacobi-symbol(words - 2, 2**exp - 1)
int jacobi(u32 exp, const std::vector<u32>& words);

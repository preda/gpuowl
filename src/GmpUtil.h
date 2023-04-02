// Copyright (C) Mihai Preda.

#pragma once

#include "common.h"
#include <gmpxx.h>

#include <string>
#include <vector>

// return GCD(bits - sub, 2^exp - 1) as a decimal string if GCD!=1, or empty string otherwise.
std::string GCD(u32 exp, const std::vector<u32>& words, u32 sub = 0);

// Represent mpz value as vector of bits with the most significant bit first.
vector<bool> bitsMSB(const mpz_class& a);

vector<bool> powerSmoothBE(u32 exp, u32 B1);
vector<bool> powerSmoothLE(u32 exp, u32 B1, u32 blockSize = 1);

// Bitlen of powerSmooth
u32 powerSmoothBits(u32 exp, u32 B1);

// Returns jacobi-symbol(words, 2**exp - 1)
int jacobi(u32 exp, const std::vector<u32>& words);

inline mpz_class mpz64(u64 h) {
  mpz_class ret{u32(h >> 32)};
  ret <<= 32;
  ret += u32(h);
  return ret;
}

// Split the "k" of a P-1 factor of the form 2*k*exponent +1 into subfactors.
vector<u32> factorize(const string& str, u32 exponent, u32 B1, u32 B2);

double log2(const string& str);

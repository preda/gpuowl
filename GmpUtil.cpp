// Copyright (C) Mihai Preda.

#include "GmpUtil.h"

#include <gmp.h>
#include <cmath>
#include <cassert>

using namespace std;

namespace {

mpz_class mpz(const vector<u32>& words) {
  mpz_class b{};
  mpz_import(b.get_mpz_t(), words.size(), -1 /*order: LSWord first*/, sizeof(u32), 0 /*endianess: native*/, 0 /*nails*/, words.data());
  return b;
}

mpz_class primorial(u32 p) {
  mpz_class b{};
  mpz_primorial_ui(b.get_mpz_t(), p);
  return b;
}

mpz_class powerSmooth(u32 exp, u32 B1) {
  if (!B1) { return 0; }
  
  mpz_class a{exp};
  a *= 256;  // boost 2s.
  for (int k = log2(B1); k >= 1; --k) { a *= primorial(pow(B1, 1.0 / k)); }
  return a;
}

u32 sizeBits(mpz_class a) { return mpz_sizeinbase(a.get_mpz_t(), 2); }

}

double log2(const string& str) {
  mpz_class n{str};
  long int e = 0;
  double d = mpz_get_d_2exp(&e, n.get_mpz_t());
  return e + log2(d);
}

u32 powerSmoothBits(u32 exp, u32 B1) {
  if (!B1) { return 0; }
  
  // Could be implemented more efficiently by summing log2() over primes.
  return sizeBits(powerSmooth(exp, B1));  
}

vector<bool> bitsBE(const mpz_class& a) {
  vector<bool> bits;
  int nBits = sizeBits(a);
  bits.reserve(nBits);
  for (int i = nBits - 1; i >= 0; --i) { bits.push_back(mpz_tstbit(a.get_mpz_t(), i)); }
  assert(int(bits.size()) == nBits);
  return bits;
}

static vector<bool> bitsLE(const mpz_class& a, u32 blockSize = 1) {
  u32 nBits = sizeBits(a);
  assert(nBits);
  u32 fillBits = blockSize - 1 - (nBits - 1) % blockSize;

  vector<bool> bits;
  bits.reserve(nBits + fillBits);
  bits.resize(fillBits);

  for (u32 i = 0; i < nBits; ++i) { bits.push_back(mpz_tstbit(a.get_mpz_t(), i)); }

  assert(bits.size() % blockSize == 0);
  return bits;
}

// return GCD(bits - sub, 2^exp - 1) as a decimal string if GCD!=1, or empty string otherwise.
std::string GCD(u32 exp, const std::vector<u32>& words, u32 sub) {
  mpz_class w = mpz(words);
  if (w == 0 || w == sub) {
    throw std::domain_error("GCD invalid input");
  }
  mpz_class resultGcd = gcd((mpz_class{1} << exp) - 1, w - sub);
  return (resultGcd == 1) ? ""s : resultGcd.get_str();
}

// MSB: Most Significant Bit first (at index 0).
vector<bool> powerSmoothBE(u32 exp, u32 B1) { return bitsBE(powerSmooth(exp, B1)); }

vector<bool> powerSmoothLE(u32 exp, u32 B1, u32 blockSize) { return bitsLE(powerSmooth(exp, B1), blockSize); }

int jacobi(u32 exp, const std::vector<u32>& words) {
  assert(!words.empty());
  mpz_class w = mpz(words);
  mpz_class m = (mpz_class{1} << exp) - 1;
  return mpz_jacobi(w.get_mpz_t(), m.get_mpz_t());
}

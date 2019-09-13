// Copyright Mihai Preda

#include "Pm1Plan.h"

#include <string>
#include <vector>
#include <numeric>
#include <array>
#include <cassert>
#include <algorithm>
#include <initializer_list>
#include <bitset>

using namespace std;

static constexpr const u32 D = 30030;
static_assert(D == 2*3*5*7*11*13);
static constexpr const u32 J = 2880;
static_assert(J == 1*2*4*6*10*12 / 2);

// Simple Erathostene's sieve.
static vector<bool> primeBits(u32 B1, u32 B2) {
  assert(B1 < B2);

  vector<bool> bits(B2 + 1);
  bits[0] = bits[1] = true;
  for (u32 p = 0; p <= B2; ++p) {
    while (p <= B2 && bits[p]) { ++p; }

    if (p > B2) { break; }

    if (p <= B1) { bits[p] = true; }

    if (u64(p) * p <= B2) { for (u32 i = p * p; i <= B2; i += p) { bits[i] = true; }}
  }
  bits.flip();
  return bits;
}

// 'value' is a small multiple of a prime: B1 < prime <= B2. Return that prime.
static u32 basePrime(u32 B1, u32 value) {
  for (u32 k : {17, 19, 23, 29, 31, 37, 41, 43, 47}) {
    if (value < B1 * k) { return value; }
    if (value % k == 0) { return value / k; }
  }
  return value;
}

template<typename T>
u32 sum(const T &v) { return accumulate(v.begin(), v.end(), 0); }

static u32 blockFor(u32 B1) { return (B1 + D / 2) / D; }

class PrimeBits {
public:
  vector<bool> bits;
  u32 B1, B2;

public:
  PrimeBits(u32 B1, u32 B2) : bits(primeBits(B1, B2)), B1(B1), B2(B2) {}
  
  void expand() { for (u32 p = B1 + 1; p <= B2 / 17; ++p) { if (bits[p]) { set(p, true); }}}

  void set(u32 value, bool what) {
    assert(value <= B2 && bits[value]);
    u32 p = basePrime(B1, value);
    bits[p] = what;

    for (u32 k : {17, 19, 23, 29, 31, 37, 41, 43, 47}) {
      u32 value = p * k;
      if (value > B2) { break; }
      bits[value] = what;
    }
  }

  template<typename T>
  vector<bitset<J>> select(const T &cond) {
    u32 beginBlock = blockFor(B1);
    u32 endBlock   = blockFor(B2 - 1) + 1;
    auto jset = getJset();

    vector<bitset<J>> ret;

    for (u32 block = beginBlock; block < endBlock; ++block) {
      bitset<J> blockBits;
      u32 pos = 0;
      for (u32 j : jset) {
        assert(j >= 1 && j < D / 2);
        u32 a = block * D + j;
        u32 b = block * D - j;
        bool onA = a <= B2 && bits[a];
        bool onB = b <= B2 && bits[b];
        if (cond(onA, onB)) {
          blockBits[pos] = true;
          if (onA) { set(a, false); }
          if (onB) { set(b, false); }
        }
        ++pos;
      }
      ret.push_back(std::move(blockBits));
    }
    return ret;
  }

  u32 size() const { return sum(bits); }
};

vector<bool> boolOr(const vector<bool> &a, const vector<bool> &b) {
  assert(a.size() == b.size());
  vector<bool> ret = a;
  for (u32 i = 0; i < b.size(); ++i) {
    if (b[i]) { ret[i] = true; }
  }
  return ret;
}

static u32 countSum(const vector<bitset<2880>>& v) {
  return accumulate(v.begin(), v.end(), 0, [](u32 a, const auto& b) { return a + b.count(); });
}

tuple<u32, u32, vector<bitset<2880>>> makePm1Plan(u32 B1, u32 B2) {
  // poor man's cache
  static u32 cacheB1 = 0, cacheB2 = 0;
  static tuple<u32, u32, vector<bitset<2880>>> cachePlan;

  if (B1 == cacheB1 && B2 == cacheB2) { return cachePlan; }
  
  PrimeBits bits(B1, B2);
  u32 nPrimes = bits.size();
  bits.expand();
  u32 nExpanded = bits.size();

  vector<bitset<2880>> doubles = bits.select([](bool a, bool b) { return a && b; });
  u32 nDoubles = countSum(doubles);
  u32 leftAfterDoubles = bits.size();

  vector<bitset<2880>> singles = bits.select([](bool a, bool b) { return a || b; });
  u32 nSingles = countSum(singles);

  vector<bitset<2880>> selected;
  transform(doubles.begin(), doubles.end(), singles.begin(), back_inserter(selected),
          [](const auto& a, const auto& b){ return a | b; });

  u32 total = countSum(selected);

  log("P-1 (B1=%u, B2=%u, D=%u): primes %u, expanded %u, doubles %u (left %u), singles %u, total %u (%.0f%%)\n",
      B1, B2, D, nPrimes, nExpanded, nDoubles, leftAfterDoubles, nSingles, total, total / float(nPrimes) * 100);

  assert(total == nDoubles + nSingles);
  assert(bits.size() == 0); // all primes covered.

  u32 beginBlock = blockFor(B1);

  cachePlan = {beginBlock, total, selected};
  cacheB1 = B1;
  cacheB2 = B2;
  return cachePlan;
};

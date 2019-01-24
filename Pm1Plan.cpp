// Copyright Mihai Preda

#include "Pm1Plan.h"

#include <string>
#include <vector>
#include <numeric>
#include <array>
#include <cassert>
#include <algorithm>
#include <initializer_list>

using namespace std;

// Simple Erathostene's sieve.
static vector<bool> primeBits(u32 B1, u32 B2) {
  assert(B1 > 0 && B1 < B2);
  
  vector<bool> bits(B2 + 1);
  bits[0] = bits[1] = true;
  for (u32 p = 0; p <= B2; ++p) {
    while (p <= B2 && bits[p]) { ++p; }
    
    if (p > B2) { break; }
    
    if (p <= B1) { bits[p] = true; }

    if (u64(p) * p <= B2) { for (u32 i = p * p; i <= B2; i += p) { bits[i] = true; } }
  }
  bits.flip();
  return bits;
}

// assert((D >= 2310 && (D % 2310 == 0)) || (D % 210 == 0));
bool isRelPrime(u32 D, u32 j) { return j%2 && j%3 && j%5 && j%7 && (D < 2310 || j%11); }

// JSet : the values 1 <= x < D/2 where GCD(x, D) == 1
static vector<u32> getJset(u32 D) {
  assert((D >= 2310 && D % 2310 == 0) || (D % 210 == 0));
  
  vector<u32> jset;
  for (u32 j = 1; j < D / 2; ++j) { if (isRelPrime(D, j)) { jset.push_back(j); } }
  return jset;
}

// 'value' is a small multiple of a prime: B1 < prime <= B2. Return that prime.
static u32 basePrime(u32 B1, u32 value) {
  for (u32 k : {11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}) {
    if (value < B1 * k) { return value; }
    if (value % k == 0) { return value / k; }
  }
  return value;
}

template<typename T> u32 sum(const T& v) { return accumulate(v.begin(), v.end(), 0); }

class PrimeBits {
  vector<bool> bits;
  u32 D;
  u32 B1, B2;
  using list_u32 = initializer_list<u32>;
  
public:
  PrimeBits(u32 D, u32 B1, u32 B2): bits(primeBits(B1, B2)), D(D), B1(B1), B2(B2) { }

  void expand() { for (u32 p = B1 + 1; p <= B2 / 11; ++p) { if (bits[p]) { set(p, true); } } }
  
  void set(u32 value, bool what) {
    assert(value <= B2 && bits[value]);
    u32 p = basePrime(B1, value);
    bits[p] = what;
  
    for (u32 k : D >= 2310
           ? list_u32{13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
           : list_u32{11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}) {
      u32 value = p * k;
      if (value > B2) { break; }
      bits[value] = what;
    }
  }

  template<typename T>
  vector<bool> select(const T& cond) {
    u32 firstBlock = (B1 + 1 + (D / 2)) / D;
    u32 lastBlock  = (B2 + (D / 2)) / D;
    assert(firstBlock > 0);
    vector<u32> jset = getJset(D);
    vector<bool> ret((lastBlock - firstBlock + 1) * (D / 4));

    for (u32 i = lastBlock; i >= firstBlock; --i) {
      for (u32 j : jset) {
        assert(j >= 1 && j < D/2);
        u32 a = i * D + j;
        u32 b = i * D - j;
        bool onA = a <= B2 && bits[a];
        bool onB = b <= B2 && bits[b];
        if (cond(onA, onB)) {
          assert(j > 0);
          ret.at((i - firstBlock) * (D / 4) + (j - 1) / 2) = true;
          if (onA) { set(a, false); }
          if (onB) { set(b, false); }
        }
      }
    }
    return ret;
  }

  u32 size() const { return sum(bits); }
};

vector<bool> boolOr(const vector<bool>& a, const vector<bool>& b) {
  assert(a.size() == b.size());
  vector<bool> ret = a;
  for (u32 i = 0; i < b.size(); ++i) {
    if (b[i]) { ret[i] = true; }
  }
  return ret;
}

static vector<bool> getBlock(const vector<bool> &all, u32 blockSize, u32 idx) {
  return vector<bool>(next(all.begin(), idx * blockSize), next(all.begin(), (idx + 1) * blockSize));
}
  
pair<u32, vector<vector<bool>>> makePm1Plan(u32 D, u32 B1, u32 B2) {
  PrimeBits bits(D, B1, B2);
  u32 nPrimes = bits.size();
  bits.expand();
  u32 nExpanded = bits.size();
  
  vector<bool> doubles = bits.select([](bool a, bool b) { return a && b; });
  u32 nDoubles = sum(doubles);
  u32 leftAfterDoubles = bits.size();
  
  vector<bool> singles = bits.select([](bool a, bool b) { return a || b; });
  u32 nSingles = sum(singles);

  vector<bool> selected = boolOr(doubles, singles);

  u32 total = sum(selected);
  log("P-1 (B1=%u, B2=%u, D=%u): primes %u, expanded %u, doubles %u (left %u), singles %u, total %u (%.0f%%)\n",
      B1, B2, D, nPrimes, nExpanded, nDoubles, leftAfterDoubles, nSingles, total, total / float(nPrimes) * 100);

  assert(bits.size() == 0); // all primes covered.
  
  u32 firstBlock = (B1 + 1 + (D / 2)) / D;
  u32 lastBlock  = (B2 + (D / 2)) / D;
  u32 blockSize  = D / 4;
  
  u32 firstNotEmpty = firstBlock;
  while (sum(getBlock(selected, blockSize, firstNotEmpty - firstBlock)) == 0) { ++firstNotEmpty; }

  log("P-1 initial %u blocks skipped (%u to %u)\n", firstNotEmpty - firstBlock, firstBlock, firstNotEmpty);

  vector<vector<bool>> blocks;
  for (u32 i = firstNotEmpty; i <= lastBlock; ++i) { blocks.push_back(getBlock(selected, blockSize, i - firstBlock)); }
  
  return {firstNotEmpty, blocks};
};

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

// JSet : the values 1 <= x < D/2 where GCD(x, D) == 1
static vector<u32> getJset(u32 D) {
  assert((D >= 2310 && D % 2310 == 0) || (D % 210 == 0));
  
  vector<u32> jset;
  for (u32 j = 1; j < D / 2; ++j) { if (j%2 && j%3 && j%5 && j%7 && (D < 2310 || j%11)) { jset.push_back(j); } }
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
  PrimeBits(u32 D, u32 B1, u32 B2): bits(primeBits(B1, B2)), D(D), B1(B1), B2(B2) {
    // cout << "Primes " << sum(bits) << endl;
    for (u32 p = B1 + 1; p <= B2 / 11; ++p) { if (bits[p]) { set(p, true); } }
    // cout << "Expanded " << sum(bits) << endl;
  }

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
  vector<pair<u32, u32>> select(const vector<u32>& jset, const T& cond) {
    vector<pair<u32, u32>> ret;
    u32 firstBlock = (B1 + 1 + (D / 2)) / D;
    u32 lastBlock  = (B2 + (D / 2)) / D;
    assert(firstBlock > 0);
    for (u32 d = lastBlock; d >= firstBlock; --d) {
      for (u32 jpos = 0; jpos < jset.size(); ++jpos) {
        u32 a = d * D + jset[jpos];
        u32 b = d * D - jset[jpos];
        bool onA = a <= B2 && bits[a];
        bool onB = b <= B2 && bits[b];
        if (cond(onA, onB)) {
          ret.emplace_back(d, jpos);
          if (onA) { set(a, false); }
          if (onB) { set(b, false); }
        }
      }
    }
    return ret;
  }

  u32 size() const { return sum(bits); }
};

Pm1Plan makePm1Plan(u32 D, u32 B1, u32 B2) {
  PrimeBits bits(D, B1, B2);

  vector<u32> jset = getJset(D);
  
  auto doubles = bits.select(jset, [](bool a, bool b) { return a && b; }); 
  
  // cout << "Doubles " << doubles.size() << endl;

  vector<pair<u32, u32>> total = bits.select(jset, [](bool a, bool b) { return a || b; });
  total.insert(total.end(), doubles.begin(), doubles.end());
  
  // cout << "Total " << total.size() << endl;
  // cout << "Left " << bits.size() << endl;    
  sort(total.begin(), total.end());

  vector<vector<u32>> blockJpos;
  
  vector<u32> acc;
  assert(!total.empty());
  u32 firstD = total.front().first;
  u32 prevD = firstD;
  for (const auto& [d, j] : total) {
    if (d != prevD) {
      blockJpos.push_back(move(acc));
      acc.clear();
      prevD = d;
    }
    acc.push_back(j);
  }
  blockJpos.push_back(move(acc));
  return Pm1Plan{move(jset), firstD, move(blockJpos)};
  
  // auto &front = total.front();
  // cout << front.first << " " << front.second << endl;

  // u32 firstBlock = (B1 + (D / 2)) / D;
  // u32 lastBlock  = (B2 + (D / 2)) / D;
  // cout << firstBlock << " " << lastBlock << " " << (lastBlock - front.first) << endl;
  // return total;
};

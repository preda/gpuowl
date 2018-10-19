#include "common.h"

#include <vector>
#include <unordered_set>

class Primes {
  vector<u32> primes;
  unordered_set<u32> primeSet;

public:
  struct Range {
    typedef vector<u32>::const_iterator T;
    T b, e;
    T begin() { return b; }
    T end() { return e; }
  };

  Primes(u32 end);

  bool isPrime(u32 x) { return primeSet.count(x); }

  Range from(u32 p) {
    auto it = primes.cbegin(), end = primes.cend();
    while (it < end && *it < p) { ++it; }
    return {it, end};
  }

  vector<pair<u32, u32>> factors(u32 x);

  vector<u32> simpleFactors(u32 x) {
    vector<u32> ret;
    for (auto p : factors(x)) { ret.push_back(p.first); }
    return ret;
  }

  vector<u32> divisors(u32 x);

  u32 zn2(u32 p);
};

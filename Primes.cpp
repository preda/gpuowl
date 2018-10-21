#include "Primes.h"

#include <cassert>
#include <cstdio>
#include <algorithm>

static u32 pow32(u32 x, int n) {
  u64 r = 1;
  for (int i = 0; i < n; ++i) { r *= x; }
  return r;
}

template<u32 P> static int order(u32 &x) { int c = 0; while (x % P == 0) { ++c; x /= P; }; return c; }
static int order(u32 &x, u32 P) { int c = 0; while (x % P == 0) { ++c; x /= P; }; return c; }
static u32 modExp2(u32 p, u32 e) {
  u32 r = 2;
  for (int i = 30 - __builtin_clz(e); i >= 0; --i) {
    r = ((e & (1u << i)) ? 2u : 1u) * u64(r) * r % p;
  }
  return r;
}

static vector<u32> genPrimes(u32 end) {
  vector<u32> primes;
  primes.push_back(2);
  vector<bool> notPrime(end / 2);
  notPrime[0] = true;
  u32 last = 0;
  while (true) {
    u32 p = last + 1;
    while (p < end && notPrime[p]) { ++p; }
    u32 prime = 2 * p + 1;
    if (prime > end) { break; }
    primes.push_back(prime);
    
    last = p;
    notPrime[p] = true;
    u64 s = 2 * u64(p) * (p + 1);
    for (u32 i = s; s < end/2 && i < end/2; i += prime) { notPrime[i] = true; }
  }
  // fprintf(stderr, "Generated %lu primes: [%u, %u]\n", primes.size(), primes.front(), primes.back());
  return primes;
}

Primes::Primes(u32 end) :
  primes(genPrimes(end)),
  primeSet(primes.begin(), primes.end())
{}

vector<pair<u32, u32>> Primes::factors(u32 x) {
  vector<pair<u32, u32>> ret;
  if (x <= 1) { return ret; }
  
  if (isPrime(x)) { ret.push_back(make_pair(x, 1u)); return ret; }
  
  if (int n = order<2>(x)) {
    ret.push_back(make_pair(2, n));
    if (isPrime(x)) { ret.push_back(make_pair(x, 1u)); return ret; }
    if (x == 1) { return ret; }
  }
  
  if (int n = order<3>(x)) {
    ret.push_back(make_pair(3, n));
    if (isPrime(x)) { ret.push_back(make_pair(x, 1u)); return ret; }
    if (x == 1) { return ret; }
  }
  
  if (int n = order<5>(x)) {
    ret.push_back(make_pair(5, n));
    if (isPrime(x)) { ret.push_back(make_pair(x, 1u)); return ret; }
    if (x == 1) { return ret; }
  }
  
  for (auto p : from(7)) {
    if (int n = order(x, p)) {
      ret.push_back(make_pair(p, n));
      if (isPrime(x)) { ret.push_back(make_pair(x, 1u)); return ret; }
      if (x == 1) { return ret; }
    }
  }
  // fprintf(stderr, "No factors for %u\n", x);
  assert(false);
  return ret;
}

vector<u32> Primes::divisors(u32 x) {
  auto f = factors(x);
  int nf = f.size();
  vector<u32> divs;
  vector<u32> count(nf);
  while (true) {
    int i = 0;
    while (i < nf && count[i] == f[i].second) {
      count[i] = 0;
      ++i;
    }
    if (i >= nf) { break; }
    ++count[i];
    u32 d = 1;
    for (int i = 0; i < nf; ++i) { d *= pow32(f[i].first, count[i]); }
    divs.push_back(d);
  }
  sort(divs.begin(), divs.end());
  return divs;
}

u32 Primes::zn2(u32 p) {
  u32 d = p - 1;
  while (d%2==0 && modExp2(p, d/2)==1) { d /= 2; }
  while (d%3==0 && modExp2(p, d/3)==1) { d /= 3; }
  
 again:
  for (u32 f : simpleFactors(d)) {
    if (f!=2 && f!=3 && modExp2(p, d/f)==1) {
      d /= f;
      goto again;
    }
  }
  
  return d;
}

#include "Pm1Plan.h"

#include <tuple>
#include <array>
#include <cassert>
#include <numeric>

vector<bool> Pm1Plan::sieve(u32 B1, u32 B2) {
  assert(2 < B1 && B1 < B2);
  vector<bool> bits(B2 + 1);
  bits[0] = bits[1] = true;
  for (u32 p = 0; p <= B2; ++p) {
    while (p <= B2 && bits[p]) { ++p; }

    if (p > B2) { break; }

    if (p <= B1) { bits[p] = true; }

    if (p < (1u << 16) && p * p <= B2) { for (u32 i = p * p; i <= B2; i += p) { bits[i] = true; }}
  }
  bits.flip();
  return bits;
}

namespace {

template<u32 D> constexpr bool isRelPrime(u32 j);
template<> constexpr bool isRelPrime<210>(u32 j)  { return j % 2 && j % 3 && j % 5 && j % 7; }
template<> constexpr bool isRelPrime<330>(u32 j)  { return j % 2 && j % 3 && j % 5          && j % 11; }
template<> constexpr bool isRelPrime<462>(u32 j)  { return j % 2 && j % 3          && j % 7 && j % 11; }
template<> constexpr bool isRelPrime<770>(u32 j)  { return j % 2          && j % 5 && j % 7 && j % 11; }
template<> constexpr bool isRelPrime<2310>(u32 j) { return j % 2 && j % 3 && j % 5 && j % 7 && j % 11; }

template<typename T>
u32 sum(const T &v) { return accumulate(v.begin(), v.end(), 0); }

// Repeatedly divides "pos" by "F" while it can, keeping it above B1.
template<u32 F>
u32 reduce(u32 B1, u32 pos) {
  while (pos > B1 * F && pos % F == 0) { pos /= F; }
  return pos;
}

constexpr u32 firstMissingFactor(u32 D) {
  switch (D) {
  case 210:
  case 420:
    return 11;
      
  case 330:
  case 660:
    return 7;
      
  case 462:
  case 2*462:
    return 5;
      
  case 770:
  case 2*770:
    return 3;
      
  case 2310:
    return 13;
  }
  assert(false);
}

// Repeatedly divides "pos" by the smallest missing factor of "D".
u32 reduce(u32 D, u32 B1, u32 pos) {
  switch (D) {
  case 210:
  case 420:
    return reduce<11>(B1, pos);
      
  case 330:
  case 660:
    return reduce<7>(B1, pos);
      
  case 462:
  case 2*462:
    return reduce<5>(B1, pos);
      
  case 770:
  case 2*770:
    return reduce<3>(B1, pos);
      
  case 2310:
    return reduce<13>(B1, pos);
  }
  assert(false);
}

// Returns whether GCD(D, j)==1.
bool isRelPrime(u32 D, u32 j) {
  switch (D) {
  case 210:
  case 2*210:
    return isRelPrime<210>(j);
  
  case 330:
  case 2*330:
    return isRelPrime<330>(j);
    
  case 462:
  case 2*462:
    return isRelPrime<462>(j);
    
  case 770:
  case 2*770:
    return isRelPrime<770>(j);
    
  case 2310:
    return isRelPrime<2310>(j);
  }
  assert(false);
}

}

// Returns the minimal number of buffers J needed for D.
u32 Pm1Plan::minBufsFor(u32 D) {
  switch (D) {
  case 210: return 24;
  case 2*210: return 2*24;
  case 330: return 40;
  case 2*330: return 2*40;
  case 462: return 60;
  case 2*462: return 2*60;
  case 770: return 120;
  case 2*770: return 2*120;
  case 2310: return 240;
  }
  assert(false);
}

u32 Pm1Plan::reduce(u32 pos) const { return ::reduce(D, B1, pos); }
template<u32 F> u32 Pm1Plan::reduce(u32 pos) const { return ::reduce<F>(B1, pos); }

Pm1Plan::Pm1Plan(u32 D, u32 nBuf, u32 B1, u32 B2, vector<bool>&& primeBits)
  : nBuf{nBuf}, primeBits{std::move(primeBits)}, D{D ? D : (nBuf >= minBufsFor(330) ? 330 : 210)}, B1{B1}, B2{B2}, jset{makeJset()} {
  nBuf = min(nBuf, MAX_BUFS);
  // log("D=%u, J=%u, nBuf=%u\n", D, minBufsFor(D), nBuf);
  
  assert(nBuf >= 24);
  assert(nBuf >= minBufsFor(D));
  assert(B1 < B2);

  // Extend the primeBits vector with guard zero bits, so that makePlan() does not read past-the-end.
  for (u32 i = 0; i < jset.back() + 1; ++i) { this->primeBits.push_back(false); }
}

Pm1Plan::Pm1Plan(u32 D, u32 nBuf, u32 B1, u32 B2) : Pm1Plan{D, nBuf, B1, B2, sieve(B1, B2)} {
}

vector<u32> Pm1Plan::makeJset() {
  vector<u32> jset;
  assert(nBuf >= minBufsFor(D));
  for (u32 j = 1; jset.size() < nBuf; j += 2) {
    if (isRelPrime(D, j)) { jset.push_back(j); }
  }
  return jset;
}

u32 Pm1Plan::primeAfter(u32 b) const {
  while (!primeBits[++b]);
  return b;
}

u32 Pm1Plan::primeBefore(u32 b) const {
  while (!primeBits[--b]);
  return b;
}

u32 Pm1Plan::getStartBlock(u32 doneB2) const {
  // "doneB2" was already done, so start with the prime following it.
  return lowerBlock(primeAfter(doneB2));
}

// Returns the prime hit by "a", or 0.
u32 Pm1Plan::hit(const vector<bool>& primes, u32 a) {
  u32 r = 0;
  return primes[r=a]
    || primes[r=reduce(a)]
    || primes[r=reduce<13>(a)]
    || primes[r=reduce<17>(a)]
    || primes[r=reduce<19>(a)]
    || primes[r=reduce<23>(a)]
    // || primes[r=reduce<29>(a)]
    ? r : 0;
}

// The largest block that covers "b" (the initial block).
u32 Pm1Plan::lowerBlock(u32 b) const { return (b + jset.back()) / D; }
  
// The smallest block that cover "b" (the final block).
u32 Pm1Plan::upperBlock(u32 b) const { return (b - jset.back()) / D + 1; }

vector<Pm1Plan::BitBlock> Pm1Plan::makePlan(PlanStats* stats) {
  // In the unlikely case that either B1 or B2 is prime:
  // B1 was included in P1, so excluded in P2.
  // B2 is included in P2.

  u32 lastPrime = primeBefore(B2 + 1);
  u32 lastBlock   = upperBlock(lastPrime);
  u32 lastCovered = lastBlock * D + jset.back();

  // All primes <= cutValue can be transposed by mulitplying with firstMissingFactor.
  u32 cutValue = lastCovered / firstMissingFactor(D);
  u32 startValue = max(primeAfter(B1), primeAfter(cutValue));
  
  u32 beginBlock = lowerBlock(startValue);
  // log("cut %u beginBlock %u\n", startValue, beginBlock);

  u32 endBlock = lastBlock + 1;

  assert(beginBlock < endBlock);

  auto primes = primeBits; // use a copy in which we'll clear the primes as we cover them.
  
  const u32 nPrimes = sum(primes);

  vector<BitBlock> selected(endBlock);

  u32 nPair = 0;      // Simple pairing: both points at block*D +/- j are primes.
  u32 nFancyPair = 0; // Fancy pairing: of block*D +/- j, one point is prime and the other is a multiple of a prime.

  // Iterating down improves (a tiny bit) pairing vs. iterating up, by extracting more fancy pairs.
  for (u32 block = endBlock - 1; block >= beginBlock; --block) {
    BitBlock& blockBits = selected[block];
      
    for (u32 pos = 0, end = jset.size(); pos < end; ++pos) {
      // We check each pair of values (a,b) that can be covered with one MUL.
      u32 j = jset[pos];
      u32 a = block * D - j;
      u32 b = block * D + j;

      if (u32 p1 = 0, p2 = 0; (p1 = hit(primes, a)) && (p2 = hit(primes, b))) {
        assert(p1 && p2 && (p1 != p2));
        // log("%u %u\n", p1, p2);
        ++nPair;
        assert(primes.at(p1) && primes.at(p2) && (p1 != p2));
        primes[p1] = primes[p2] = false;
        assert(!blockBits[pos]);
        blockBits[pos] = true;
      }
    }
  }
  
  u32 nSingle = 0;  // Leftover primes that could not be paired (single).
  // The block iteration order does not matter for singles.
  for (u32 block = beginBlock; block < endBlock; ++block) {
    BitBlock& blockBits = selected[block];
    
    for (u32 pos = 0, end = jset.size(); pos < end; ++pos) {
      u32 j = jset[pos];
      u32 a = block * D - j;
      u32 b = block * D + j;

      if (u32 p1 = 0, p2 = 0; (p1 = hit(primes, a)) || (p2 = hit(primes, b))) {
        assert(!(p1 && p2) && (p1 || p2) && (p1 != p2));
        ++nSingle;
        assert((primes.at(p1) || primes.at(p2)) && !(primes[p1] && primes[p2]));
        primes[p1] = primes[p2] = false;
        assert(!blockBits[pos]);
        blockBits[pos] = true;
      }
    }
  }

  assert(sum(primes) == 0);  // all primes covered.
  // log("%u %u %u %u\n", nPair, nFancyPair, nSingle, nPrimes);
  assert((nPair + nFancyPair) * 2 + nSingle == nPrimes);  // the counts agree with the number of primes.

  u32 nBlocks = endBlock - beginBlock;

  // The block transition cost is approximated as 2 MULs.
  float cost = nPair + nFancyPair + nSingle + 2 * nBlocks;
  float percentPaired = 100 * (1 - nSingle / float(nPrimes));

  u32 firstPrime = primeAfter(B1);
  log("D=%u: %u primes in [%u, %u]: cost %.2fM (pair: %u + %u, single: %u, (%.0f%% paired), blocks: %u)\n",
      D, nPrimes, firstPrime, lastPrime,
      cost * (1.0f / 1'000'000),
      nPair, nFancyPair, nSingle, percentPaired, nBlocks);

  if (stats) {
    *stats = {nPrimes, firstPrime, lastPrime, cost, {nPair + nFancyPair}, nSingle, nBlocks};
  }
  
  return selected;
}

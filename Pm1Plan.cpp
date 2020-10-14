#include "Pm1Plan.h"
#include "Args.h"

#include <tuple>
#include <array>
#include <cassert>
#include <numeric>

// Simple Erathostene's sieve restricted to the range [B1, B2].
// Returns a vector of bits.
// Only the bits between B1 and B2 are set where there is a prime.
vector<bool> sieve(u32 B1, u32 B2) {
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

// Returns the minimal number of buffers J needed for D.
u32 JforD(u32 D) {
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

// Repeatedly divides "pos" by "F" while it can, keeping it above B1.
template<u32 F>
u32 reduce(u32 B1, u32 pos) {
  while (pos > B1 * F && pos % F == 0) { pos /= F; }
  return pos;
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

u32 Pm1Plan::reduce(u32 pos) const { return ::reduce(D, B1, pos); }
template<u32 F> u32 Pm1Plan::reduce(u32 pos) const { return ::reduce<F>(B1, pos); }

Pm1Plan::Pm1Plan(const Args& args, u32 nBuf, u32 B1, u32 B2)
  : nBuf{nBuf}, D{args.D ? args.D : (nBuf >= JforD(330) ? 330 : 210)}, B1{B1}, B2{B2}, jset{makeJset()}, primeBits{makePrimeBits()} {
  nBuf = min(nBuf, MAX_BUFS);
  log("D=%u, J=%u, nBuf=%u\n", D, JforD(D), nBuf);
    
  assert(nBuf >= 24);
  assert(nBuf >= JforD(D));
  assert(B1 < B2);
}

vector<bool> Pm1Plan::makePrimeBits() {
  vector<bool> primes = sieve(B1, B2);
  
  // Extend the primeBits vector with guard zero bits, so that makePlan() does not read past-the-end.
  for (u32 i = 0; i < jset.back() + 1; ++i) { primes.push_back(false); }

  return primes;
}

vector<u32> Pm1Plan::makeJset() {
  vector<u32> jset;
  assert(nBuf >= JforD(D));
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

vector<Pm1Plan::BitBlock> Pm1Plan::makePlan() {
  // In the unlikely case that either B1 or B2 is prime:
  // B1 was included in P1, so excluded in P2.
  // B2 is included in P2.
  u32 firstPrime = primeAfter(B1);
  u32 lastPrime = primeBefore(B2 + 1);
  u32 beginBlock = lowerBlock(firstPrime);
  u32 endBlock   = upperBlock(lastPrime) + 1;
  assert(beginBlock < endBlock);

  auto primes = primeBits; // use a copy in which we'll clear the primes as we cover them.
  
  const u32 nPrimes = sum(primes);

  vector<BitBlock> selected(endBlock);

  u32 nPair = 0;      // Simple pairing: both points at block*D +/- j are primes.
  u32 nFancyPair = 0; // Fancy pairing: of block*D+/-j, one point is prime and the other is a multiple of a prime.

  // Iterating down improves (a tiny bit) pairing vs. iterating up, by extracting more fancy pairs.
  for (u32 block = endBlock - 1; block >= beginBlock; --block) {
    BitBlock& blockBits = selected[block];
      
    for (u32 pos = 0, end = jset.size(); pos < end; ++pos) {

      // We check each pair of values (a,b) that can be covered with one MUL.
      u32 j = jset[pos];
      u32 a = block * D - j;
      u32 b = block * D + j;
      
      // If at least one is a prime, try to form a pair of primes.
      if (primes[a] || primes[b]) {
        if (primes[a] && primes[b]) {
          // Simple pair: they're both primes.
          ++nPair;
          primes[a] = primes[b] = false;
          blockBits[pos] = true;          
        } else {
          // Fancy pair: one is a prime, the other is a multiple of a prime.
          u32 c = !primes[a] ? a : b;
          u32 r = 0;
          if (primes[r = reduce(c)]
              || primes[r = reduce<13>(c)]
              || primes[r = reduce<17>(c)]
              || primes[r = reduce<19>(c)]) {
            ++nFancyPair;
            primes[r] = primes[a] = primes[b] = false;
            assert(!blockBits[pos]);
            blockBits[pos] = true;
          }
        }
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
      if (primes[a] || primes[b]) {
        ++nSingle;
        primes[a] = primes[b] = false;
        assert(!blockBits[pos]);
        blockBits[pos] = true;
      }
    }
  }

  assert(sum(primes) == 0);  // all primes covered.
  assert((nPair + nFancyPair) * 2 + nSingle == nPrimes);  // the counts agree with the number of primes.

  u32 nBlocks = endBlock - beginBlock;

  // The block transition cost is approximated as 2 MULs.
  float cost = nPair + nFancyPair + nSingle + 2 * nBlocks;
  float percentPaired = 100 * (1 - nSingle / float(nPrimes));
  
  log("D=%u: %u primes in [%u, %u]: cost %.3fM (%.1f%%) (pair: %u + %u, single: %u, (%.0f%% paired), blocks: %u)\n",
      D, nPrimes, firstPrime, lastPrime,
      cost * (1.0f / 1'000'000), cost / nPrimes * 100,
      nPair, nFancyPair, nSingle, percentPaired, nBlocks);
  return selected;
}

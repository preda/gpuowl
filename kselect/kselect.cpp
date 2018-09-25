// Copyright 2018 by Mihai Preda.

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <queue>
#include <cassert>
#include <unordered_set>
#include <unordered_map>
#include <bitset>
#include <cmath>
#include <memory>
#include <functional>
#include <tuple>

typedef unsigned u32;
typedef unsigned long long u64;

using namespace std;

// Returns all the primes p such that: p >= start and p < 2*N.
template<u32 N> vector<u32> smallPrimes(u32 start) {
  vector<u32> primes;
  if (N < 1) { return primes; }
  if (2 >= start) { primes.push_back(2); }
  u32 limit = sqrt(N);
  unique_ptr<bitset<N>> notPrime = make_unique<bitset<N>>();
  notPrime->set(0, true);
  u32 last = 0;
  while (true) {
    u32 p = last + 1;
    while (p < N && notPrime->test(p)) { ++p; }
    if (p >= N) { break; }
    last = p;
    notPrime->set(p, true);
    u32 prime = 2 * p + 1;
    if (prime >= start) { primes.push_back(prime); }
    if (p <= limit) { for (u32 i = 2 * p * (p + 1); i < N; i += prime) { notPrime->set(i, true); } }
  }
  fprintf(stderr, "Generated %lu primes %u - %u\n", primes.size(), primes.front(), primes.back());
  return primes;  
}

u32 pow32(u32 x, int n) {
  u64 r = 1;
  for (int i = 0; i < n; ++i) { r *= x; }
  return r;
}

vector<u32> primes = smallPrimes<180'000'000>(2);

unordered_set<u32> primeSet(primes.begin(), primes.end());

bool isPrime(u32 x) { return primeSet.count(x); }

struct PrimeRange {
  typedef vector<u32>::const_iterator T;
  T b, e;
  T begin() { return b; }
  T end() { return e; }
};

PrimeRange primesFrom(u32 p, u32 size = 0xffffffffu) {
  auto it = primes.cbegin(), end = primes.cend();
  while (it < end && *it < p) { ++it; }
  return {it, it + min(u32(end - it), size)};
}

template<u32 P> int order(u32 &x) { int c = 0; while (x % P == 0) { ++c; x /= P; }; return c; }
int order(u32 &x, u32 P) { int c = 0; while (x % P == 0) { ++c; x /= P; }; return c; }

vector<pair<u32, u32>> factors(u32 x) {
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
  
  for (auto p : primesFrom(7)) {
    if (int n = order(x, p)) {
      ret.push_back(make_pair(p, n));
      if (isPrime(x)) { ret.push_back(make_pair(x, 1u)); return ret; }
      if (x == 1) { return ret; }
    }
  }
  fprintf(stderr, "No factors for %u\n", x);
  assert(false);
  return ret;
}

vector<u32> divisors(u32 x) {
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

class Heap {
  typedef pair<u32, float> ValueType;
  vector<ValueType> heap;
  vector<u32> kToPos;

  void heapSwap(u32 a, u32 b) {
    if (a != b) {
      kToPos[heap[a].first] = b;
      kToPos[heap[b].first] = a;
      swap(heap[a], heap[b]);
    }
  }

  static bool greater(ValueType a, ValueType b) {
    return a.second > b.second || (a.second == b.second && a.first >= b.first);
  }
  
  void bubbleUp(u32 pos) {
    while (true) {
      if (pos == 0) { break; }
      u32 parent = (pos - 1) / 2;
      if (greater(heap[parent], heap[pos])) { break; }
      heapSwap(pos, parent);
      pos = parent;
    }
  }

  bool bubbleDown(u32 pos) {
    bool lowered = false;
    while (true) {
      u32 child = 2 * pos + 1;
      if (child >= heap.size()) { break; }
      if (child + 1 < heap.size() && greater(heap[child + 1], heap[child])) { ++child; }
      if (greater(heap[pos], heap[child])) { break; }
      heapSwap(pos, child);
      pos = child;
      lowered = true;
    }
    return lowered;
  }

  u32 getPos(u32 k) {
    u32 pos = kToPos[k];
    assert(pos < heap.size() && k == heap[pos].first);
    return pos;
  }

  void erasePos(u32 pos) {
    float x = heap.back().second;
    heapSwap(pos, heap.size() - 1);
    kToPos[heap.back().first] = 0;
    bool goUp = x >= heap.back().second;
    heap.pop_back();
    if (goUp) {
      bubbleUp(pos);
    } else {
      bubbleDown(pos);
    }
    // if (!bubbleDown(pos)) { bubbleUp(pos); }
  }
  
public:
  Heap(u32 maxK) :
    kToPos(maxK + 1)
  {
  }

  size_t size() { return heap.size(); }

  bool empty() { return heap.empty(); }
  
  pair<u32, float> top() { return heap.front(); }
  
  void push(u32 k, float value) {
    heap.push_back(make_pair(k, value));
    u32 pos = heap.size() - 1;
    kToPos[k] = pos;
    bubbleUp(pos);
  }

  bool contains(u32 k) { return kToPos[k] || (!heap.empty() && k == heap.front().first); }

  float get(u32 k) { return heap[kToPos[k]].second; }

  void decrease(u32 k, float delta, float limit) {
    assert(delta > 0);
    if (contains(k)) {
      u32 pos = getPos(k);
      float newValue = heap[pos].second - delta;
      if (newValue >= limit) {
        heap[pos].second = newValue;
        bubbleDown(pos);
      } else {
        erasePos(pos);
      }
    }
  }
};

struct Counter {
  u32 on;
  u32 total;

  void mark(bool isOn) {
    ++total;
    on += isOn;
  }
};

class Cover {
  u32 exp;
  u32 B1;
  u32 slope;
  float rslope;
  float limit;
  vector<float> base;
public:
  unique_ptr<Heap> heap;
private:
  vector<pair<u32,u32>> primes100M;
  unordered_multimap<u32, u64> zToP;

  float slopedValue(float value, u32 k) {
    return value * rslope * (slope - k);
  }

  float pSlope(u64 p, u32 slopeStart, u32 slope) {
    return (p < slopeStart) ? 1 : (float(slope) / (p - (slopeStart - slope)));
  }
  
  void read(u32 exp, u32 B1, u32 slopeStart, u32 slope) {
    u64 p = 0;
    u32 z = 0;
    double sum = 0;
    while (scanf("%llu %u\n", &p, &z) == 2) {
      if (p > B1 && z < exp) {
        float value = pSlope(p, slopeStart, slope);
        base[z] += value;
        sum += value;
        if (p < 100'000'000) { primes100M.push_back(make_pair(p, z)); }
        zToP.insert(make_pair(z, p));
      }
    }
    fprintf(stderr, "read sum %f\n", sum);
  }

  vector<u64> pCover(vector<u32> zCover) {
    vector<u64> pCover;
    for (u32 z : zCover) {
      auto [it, end] = zToP.equal_range(z);
      assert(it != end);
      for (; it != end; ++it) { pCover.push_back(it->second); }
    }
    sort(pCover.begin(), pCover.end());
    return pCover;
  }
  
public:
  Cover(u32 exp, u32 B1, u32 slope, float valueLimit, u32 pSlopeStart, u32 pSlope) :
    exp(exp),
    B1(B1),
    slope(slope),
    rslope(1.0f / slope),
    limit(valueLimit),
    base(exp)
  {

    read(exp, B1, pSlopeStart, pSlope);

    heap = make_unique<Heap>(exp);
    for (u32 k = 0; k < exp; ++k) {
      float sum = 0;
      for (u32 d : divisors(k)) { sum += base[d]; }
      float value = slopedValue(sum, k);
      if (value >= limit) {
        heap->push(k, value);
      }
      if (k && (k % 10'000'000 == 0)) { fprintf(stderr, "%2uM : %lu\n", k / 1'000'000, heap->size()); }
    }    
    fprintf(stderr, "Heap %lu\n", heap->size());
  }

  void printStats() {
    u32 step = 10'000'000;
    vector<Counter> counters(10);
    vector<u32> notCovered;
    for (auto [p, z] : primes100M) {
      bool covered = z < exp && base[z] <= 0;
      u32 c = p / step;
      assert(c < counters.size());
      counters[c].mark(covered);
      if (!covered && notCovered.size() < 20) { notCovered.push_back(p); }
    }
    
    u32 total = 0;
    for (int i = 0; i < int(counters.size()); ++i) {
      Counter c = counters[i];
      fprintf(stderr, "%7u (%6.2f%%) of %8u primes between %3uM and %3uM\n",
              c.on, c.on * 100.0f / c.total, c.total,
              ((i == 0) ? B1 : i * step) / 1000000, (i + 1) * step/1000000);
      total += c.on;
    }
    fprintf(stderr, "covered under 100M: %u\n", total);

    fprintf(stderr, "First not covered: ");
    for (u32 p : notCovered) {
      fprintf(stderr, "%u ", p);
    }
    fprintf(stderr, "\n");
  }  

  // Return: (score, vector of covered).
  pair<float, vector<u32>> remove(u32 k) {
    float sum = 0;
    vector<u32> covered;
    for (u32 d : divisors(k)) {
      if (float value = base[d]) {
        sum += value;
        covered.push_back(d);
        for (u32 multiple = d; multiple < exp; multiple += d) {
          heap->decrease(multiple, slopedValue(value, multiple), limit);
        }
      }
      base[d] = 0;
    }
    return make_pair(slopedValue(sum, k), covered);
  }
  
  tuple<u32, vector<u64>, float> popBest() {
    auto [bestK, bestScore] = heap->top();
    auto [rmScore, zCover] = remove(bestK);
    if (fabs(bestScore - rmScore) > 0.001f) {
      fprintf(stderr, "score mismatch : %u, %g, %g\n", bestK, bestScore, rmScore);
    }
    // assert(fabsf(bestScore - rmScore) < 0.0002f);
    return make_tuple(bestK, pCover(zCover), bestScore);
  }

  size_t size() { return heap->size(); }

  bool empty() { return heap->empty(); }
};

int main(int argc, char **argv) {

  int n = 800000;
  const u32 exp = 89000000;
  const u32 B1  = 1000000;
  u32 slopeStart = 17'000'000;
  u32 slope = 17'000'000;

  
  /*
  int n = 600000;
  const u32 exp = 89000000;
  const u32 B1  = 730000;
  u32 slopeStart = 14'000'000;
  u32 slope = 14'000'000;
  */

  /*
  int n = (argc > 1) ? atoi(argv[1]) : 2000000;
  const u32 exp = 332210000;
  const u32 B1  = 3000000;
  u32 slopeStart = 50'000'000;
  u32 slope      = 45'000'000;
  */
  
  Cover cover(exp, B1, exp * 1.5, 0.7, slopeStart, slope);

  fprintf(stderr, "Prime slope %u, %u\n", slopeStart, slope);
  u32 nCover = 0;
  float lastScore = 0;
  for (int i = 0; i < n && !cover.empty(); ++i) {
    auto [k, pCover, score] = cover.popBest();
    lastScore = score;
    assert(!pCover.empty());
    u32 nP = pCover.size();
    nCover += nP;
    printf("%u %u", k, nP);
    for (u64 p : pCover) { printf(" %llu", p); }
    printf("\n");
    if (i % 100000 == 0) {
      fprintf(stderr, "%2d: %8u %g; heap size %lu\n", i / 100000, k, score, cover.size());
    }
  }
  cover.printStats();
  fprintf(stderr, "Total covered: %u; last score: %f\n", nCover, lastScore);
}

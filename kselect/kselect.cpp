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
  vector<pair<u32, float>> heap;
  vector<u32> kToPos;

  void heapSwap(u32 a, u32 b) {
    if (a != b) {
      kToPos[heap[a].first] = b;
      kToPos[heap[b].first] = a;
      swap(heap[a], heap[b]);
    }
  }

  void bubbleUp(u32 pos) {
    while (true) {
      if (pos == 0) { return; }
      u32 parent = (pos - 1) / 2;
      float parentValue = heap[parent].second;
      float childValue  = heap[pos].second;
      if (parentValue > childValue
          || (parentValue == childValue && heap[parent].first >= heap[pos].first)) {
        break;
      }
      heapSwap(pos, parent);
      pos = parent;
    }
  }

  void bubbleDown(u32 pos) {
    while (true) {
      u32 child = 2 * pos + 1;
      if (child >= heap.size()) { return; }
      if (child + 1 < heap.size() && heap[child + 1].second > heap[child].second) { ++child; }
      float parentValue = heap[pos].second;
      float childValue = heap[child].second;
      if (parentValue > childValue
          || (parentValue == childValue && heap[pos].first >= heap[child].first)) {
        break;
      }
      heapSwap(pos, child);
      pos = child;
    }
  }

  u32 getPos(u32 k) {
    u32 pos = kToPos[k];
    assert(k == heap[pos].first);
    return pos;
  }

  void erasePos(u32 pos) {
    heapSwap(pos, heap.size() - 1);
    kToPos[heap.back().first] = 0;
    heap.pop_back();
    bubbleDown(pos);
  }
  
public:
  Heap(u32 maxK) :
    kToPos(maxK + 1)
  {
  }

  size_t size() { return heap.size(); }

  bool empty() { return heap.empty(); }
  
  pair<u32, float> top() { return heap.front(); }
  
  void pop() { erasePos(0); }
  
  void push(u32 k, float value) {
    heap.push_back(make_pair(k, value));
    u32 pos = heap.size() - 1;
    kToPos[k] = pos;
    bubbleUp(pos);
  }

  bool contains(u32 k) { return kToPos[k] || (!heap.empty() && k == heap.front().first); }
  
  void updateValue(u32 k, float value) {
    u32 pos = getPos(k);
    assert(value < heap[pos].second);
    heap[pos].second = value;
    bubbleDown(pos);
  }

  void erase(u32 k) { erasePos(getPos(k)); }

  void decrease(u32 k, float delta, float limit) {
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
  unique_ptr<Heap> heap;
  vector<pair<u32,u32>> primes100M;

  float slopedValue(float value, u32 k) {
    return value * rslope * (slope - k);
  }

  float pSlope(u64 p, u32 slopeStart, u32 slope) {
    return (p < slopeStart) ? 1 : (float(slope) / (p - slopeStart + slope));
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
      }
    }
    fprintf(stderr, "read sum %f\n", sum);
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
    u64 p = 0;
    u32 z = 0;
    u32 n = 31;
    u32 step = 20'000'000;
    vector<Counter> counters(5);
    for (auto pp : primes100M) {
      u32 p = pp.first;
      u32 z = pp.second;
      bool covered = z < exp && base[z] <= 0;
      u32 c = p / step;
      assert(c < 5);
      counters[c].mark(covered);
    }
    
    u32 total = 0;
    for (int i = 0; i < 5; ++i) {
      Counter c = counters[i];
      fprintf(stderr, "%7u (%6.2f%%) of %8u primes between %3uM and %3uM\n",
              c.on, c.on * 100.0f / c.total, c.total,
              ((i == 0) ? B1 : i * step) / 1000000, (i + 1) * step/1000000);
      total += c.on;
    }
    fprintf(stderr, "covered under 100M: %u\n", total);
  }  

  float remove(u32 k) {
    float sum = 0;
    for (u32 d : divisors(k)) {
      if (float value = base[d]) {
        sum += value;
        for (u32 multiple = d; multiple < exp; multiple += d) {
          heap->decrease(multiple, slopedValue(value, multiple), limit);
        }
      }
      base[d] = 0;
    }
    return slopedValue(sum, k);
  }
  
  pair<u32, float> popBest() {
    auto best = heap->top();
    u32 k = best.first;
    float s = remove(k);
    assert(abs(best.second - s) < 0.0001f);
    return best;
  }

  size_t size() { return heap->size(); }

  bool empty() { return heap->empty(); }
};

int main(int argc, char **argv) {
  int n = (argc > 1) ? atoi(argv[1]) : 1800000;
  
  const u32 exp = 88590000;
  const u32 B1  = 2000000;
  Cover cover(exp, B1, exp * 1.1, 0.5, 40'000'000, 20'000'000);

  /*
  for (u32 k : {
650179, 656479, 681301, 783911, 786881, 788971, 789703, 789877, 826367, 841831,
861829, 863201, 871511, 877199, 885673, 920281, 923053, 925693, 933119, 936601,
942511, 945473, 946819, 953401, 957211, 957577, 960049, 960898, 964331, 967649,
973561, 976597, 980131, 987953, 991297, 991351, 992287, 995734}) {
    u32 s1 = cover.size();
    float s = cover.remove(k);
    fprintf(stderr, "prune %u : score %f, size delta %u\n", k, s, u32(s1 - cover.size()));
  }
  */
  
  for (int i = 0; i < n && !cover.empty(); ++i) {
    auto pp = cover.popBest();
    u32 best = pp.first;
    float score = pp.second;
    printf("%8u %.1f\n", best, score);
    if (i % 100000 == 0) { fprintf(stderr, "%8u %.1f %lu\n", best, score, cover.size()); }    
  }

  cover.printStats();
}

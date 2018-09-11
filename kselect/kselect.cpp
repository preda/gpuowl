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

// Returns all the primes p such that: p >= start and p < 2*N; at most maxSize primes.
template<u32 N> vector<u32> smallPrimes(u32 start, u32 maxSize) {
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
    if (prime >= start) {
      primes.push_back(prime);
      if (primes.size() >= maxSize) { break; }
    }
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

auto primes = smallPrimes<180'000'000>(2, 100'000'000);

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

  /*
  void rawBubbleDown(u32 pos) {
    while (true) {
      u32 child = 2 * pos + 1;
      if (child >= heap.size()) { break; }
      if (child + 1 < heap.size() && heap[child + 1].second > heap[child].second) { ++child; }
      if (heap[pos].second >= heap[child].second) { break; }
      swap(heap[pos], heap[child]);
      pos = child;
    }
  }
  */

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

  /*
  Heap(u32 maxK, vector<pair<u32, float>> &&raw) :
    heap(raw),
    kToPos(maxK + 1)
  {
    for (u32 pos = heap.size() / 2 - 1; ; --pos) {
      rawBubbleDown(pos);
      if (pos == 0) { break; }
    }
    for (u32 pos = 0, end = heap.size(); pos < end; ++pos) {
      kToPos[heap[pos].first] = pos;
    }
  }
  */

  size_t size() { return heap.size(); }

  bool empty() { return heap.empty(); }
  
  pair<u32, float> top() { return heap.front(); }
      // make_pair(heap.front().k, heap.front().value); }
  
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

class Cover {
  u32 exp;
  u32 slope;
  float rslope;
  float limit;
  unique_ptr<Heap> heap;

  float slopedValue(float value, u32 k) {
    return value * rslope * (slope - k);
  }

  float pSlope(u64 p, u32 slopeStart, u32 slope) {
    return (p < slopeStart) ? 1 : (float(slope) / (p - slopeStart + slope));
  }
  
  void read(u32 exp, u32 B1, u32 slopeStart, u32 slope) {
    FILE *fi = fopen("p-order.txt", "rb");
    u64 p = 0;
    u32 z = 0;
    double sum = 0;
    while (fscanf(fi, "%llu %u\n", &p, &z) == 2) {
      // if (p >= pEnd) { break; } // 10000'000'000ul
      if (p > B1 && z < exp) {
        float value = pSlope(p, slopeStart, slope);
        // p < 40'000'000 ? 1 : (20'000'000.0f / (p - 20'000'000));
        // float value = float(B1 + slope) / (p + slope);
        base[z] += value;
        sum += value;
      }
    }
    fclose(fi);
    fprintf(stderr, "read sum %f\n", sum);
  }

public:
  vector<float> base;

  Cover(u32 exp, u32 slope, float valueLimit, u32 pSlopeStart, u32 pSlope) :
    exp(exp),
    slope(slope),
    rslope(1.0f / slope),
    limit(valueLimit),
    base(exp)
  {

    read(exp, 2'000'000, pSlopeStart, pSlope);

    // vector<pair<u32, float>> values;
    // values.reserve(exp);
    heap = make_unique<Heap>(exp);
    for (u32 k = 0; k < exp; ++k) {
      float sum = 0;
      for (u32 d : divisors(k)) { sum += base[d]; }
      float value = slopedValue(sum, k);
      if (value >= limit) {
        // values.push_back(make_pair(k, value));        
        heap->push(k, value);
      }
      if (k % 10'000'000 == 0) { fprintf(stderr, "%2uM : %lu\n", k / 1'000'000, heap->size()); }
    }    
    // heap = make_unique<Heap>(exp, move(values));

    fprintf(stderr, "Heap %lu\n", heap->size());
  }

  pair<u32, float> popBest() {
    auto best = heap->top();
    u32 k = best.first;
    
    for (u32 d : divisors(k)) {
      if (float value = base[d]) {
        for (u32 multiple = d; multiple < exp; multiple += d) {
          heap->decrease(multiple, slopedValue(value, multiple), limit);
        }
      }
      base[d] = 0;
    }

    return best;
  }

  size_t size() { return heap->size(); }

  bool empty() { return heap->empty(); }
};

struct Counter {
  u32 on;
  u32 total;

  void mark(bool isOn) {
    ++total;
    on += isOn;
  }
};

void coverage(u32 exp, const vector<float> &base) {
  FILE *fi = fopen("p-order.txt", "rb");

  u64 p = 0;
  u32 z = 0;
  u32 n = 31;
  u32 step = 20'000'000;
  vector<Counter> counters(n);
  while (fscanf(fi, "%llu %u\n", &p, &z) == 2) {
    if (p < 2'000'000) { continue; }
    // if (p > 200'000'000) { break; }
    bool covered = z < exp && base[z] <= 0;
    counters[min(n - 1, u32(p / step))].mark(covered);
  }
  fclose(fi);

  u32 total = 0;
  for (int i = 0; i < n; ++i) {
    Counter c = counters[i];
    fprintf(stderr, "%7u (%.2f%%) of %8u primes between %3uM and %3uM\n",
            c.on, c.on * 100.0f / c.total, c.total,
            (i == 0) ? 2 : i * step/1000000, (i == n - 1) ? 0 : (i + 1) * step/1000000);
    total += c.on;
  }
  fprintf(stderr, "big total %u\n", total);
}


int main() {
  const u32 exp = 88590000;
  Cover cover(exp, exp * 1.05, 0.4, 40'000'000, 20'000'000);

  for (int i = 0; i < 1000000 && !cover.empty(); ++i) {
    auto pp = cover.popBest();
    u32 best = pp.first;
    float score = pp.second;
    printf("%8u %.1f\n", best, score);
    if (i % 10000 == 0) { fprintf(stderr, "%8u %.1f %lu\n", best, score, cover.size()); }    
  }

  coverage(exp, cover.base);
}

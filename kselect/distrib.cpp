#include <cstdio>
#include <vector>
#include <cmath>
#include <bitset>
#include <algorithm>
#include <memory>
#include <cassert>
#include <unordered_set>

using namespace std;
typedef unsigned u32;
typedef unsigned long long u64;

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

vector<u32> pv;
vector<u32> zv;

double cover(u32 k) {
  double sum = 0;
  for (u32 d : divisors(k)) {
    if (d < zv.size() - 1) {
      u32 b = zv[d];
      u32 e = zv[d + 1];
      for (auto it = pv.begin() + b, end = pv.begin() + e; it < end; ++it) {
        if (u32 p = *it) {
          *it = 0;
          sum += 1.0 / p;
        }
      }      
    }
  }
  return sum;
}

struct Work {
  u32 n, m, gcd;
  double p;
  
  double cost() { return n + 2 * m + 2000 * gcd; }
};

int main() {
  FILE *fi = fopen("z.txt", "rb");

  u64 bigP;
  u32 p, z;
  u32 E = 90'000'000;
  u32 B1 = 1'000'000;
  double costB1 = 1.442 * B1; // approx nb. of squarings in P-1(B1).
  double pFirstStage  = 0.0166129;
  double pSecondStage = 0.03753 - pFirstStage;
    // 0.0364033 - pFirstStage;
  
  while (fscanf(fi, "%llu %u\n", &bigP, &z) == 2) {
    u32 p = u32(bigP);
    if (bigP != p) { continue; }
    if (z >= E) { break; }
    if (p <= B1) { continue; }
    while (z >= zv.size()) { zv.push_back(pv.size()); }
    assert(z == zv.size() - 1);
    pv.push_back(p);
  }
  zv.push_back(pv.size());
  
  fclose(fi);

  vector<Work> work;
  work.push_back({u32(B1 * 1.44), 0, 1u, pFirstStage});
  
  fi = stdin;
  assert(fscanf(fi, "B1=1000000\n") == 0);
  u32 lastK, prevK = 0;
  u32 k;
  double total = 0;
  double sum = 0;
  u32 muls = 0;
  vector<double> sv; // (90);
  vector<u32> nv; // (90);
  
  while (fscanf(fi, "%u", &k) == 1) {
    if (k / 1000000 != prevK / 1000000 && muls > 2000) {
      u32 n = (k / 1000000 - lastK / 1000000) * 1000000;
      work.push_back({n, muls, 1u, sum});
      total += sum;
      muls = 0;
      sum = 0;
      lastK = k;
    }
    prevK = k;
    sum += cover(k);
    ++muls;
  }
  u32 n = (k / 1000000 - lastK / 1000000) * 1000000;
  work.push_back(Work{n, muls, 1u, sum});
  total += sum;

  for (auto &w : work) { w.p *= pSecondStage / total; }
  work[0].p = pFirstStage;

  work.push_back({u32(B1 * 1.44), 0, 0, 0});
  work.push_back(Work{E, 0, 0, 0}); // double check

  for (auto w : work) {
    printf("%u %u %u %f\n", w.n, w.m, w.gcd, w.p);
  }

  double cost = 0;
  for (auto it = work.rbegin(), end = work.rend(); it != end; ++it) {
    cost = it->cost() + cost * (1 - it->p);
    printf("%f\n", cost);
  }
  printf("Cost %f\n", cost / E);

  work.clear();
  work.push_back(Work{2200000, 0, 0, 0.031});
  work.push_back(Work{E, 0, 0, 0});
  work.push_back(Work{E, 0, 0, 0});
  cost = 0;
  for (auto it = work.rbegin(), end = work.rend(); it != end; ++it) {
    cost = it->cost() + cost * (1 - it->p);
    printf("%f\n", cost);
  }
  printf("Cost %f\n", cost / E);
  
  
  /*
  
  sv.push_back(sum);
  nv.push_back(n);
  total += sum;

  printf("%u %u %u\n", k, prevK, nv.back());

  double ss = 0;
  for (u32 i = 0; i < sv.size(); ++i) {
    double s = sv[i];
    u32 n = max(nv[i], 1u);
    ss += s;
    printf("%dM : %.0f %.0f%% %g %u\n", i+1, 1000 / total * s, 100 / total * ss, s / (n), n);
  }
  */

}

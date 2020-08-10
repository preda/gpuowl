#include "pm1prob.h"
#include <cstdio>
#include <cmath>
#include <utility>
#include <tuple>

using u32 = unsigned;
using namespace std;

double nPrimesBetween(u32 B1, u32 B2) { return B2/log(B2) - B1/log(B1); }

double workForBounds(u32 B1, u32 B2) { return B1 * 1.442 * 1.1 + nPrimesBetween(B1, B2); }

pair<u32, u32> boundsFor(double ratio, double work) {
  u32 lo = 100'000;
  u32 hi = 5'000'000;
  while (true) {
    u32 B1 = (lo + hi) / 2;
    u32 B2 = B1 * ratio;
    double w = workForBounds(B1, B2);
    if (abs(w - work) / work < 0.01) { return {B1, B2}; }
    if (w < work) {
      lo = B1;
    } else {
      hi = B1;
    }
  }
}

tuple<double,u32,u32> bestGain(u32 exponent, u32 factored, double ratioB2B1) {
  double lo = 0.5 / 100;
  double hi = 6.0 / 100;
  while (true) {
    double work = (lo + hi) / 2;
    auto [B1, B2] = boundsFor(ratioB2B1, work * exponent);
    double p = pm1(exponent, factored, B1, B2);
    // fprintf(stderr, "%f %f %f\n", ratioB2B1, work, p);
    if (work >= p && (work - p) < 0.001 / 100) {
      // fprintf(stderr, "\n");
      return {p, B1, B2};
    }
    if (work >= p) {
      hi = work;
    } else {
      lo = work;
    }
  }
}

int main(int argc, char*argv[]) {
  if (argc < 2) { return 1; }
  u32 factored = atoi(argv[1]);
  
  u32 bestB1, bestB2, bestR;
  double best = 0;
  for (double r = 10; r <= 80; r += 5) {
    auto [p, B1, B2] = bestGain(100'000'000, factored, r);
    printf("%3.0f %.3f%% %u %u\n", r, p*100, B1, B2);
    if (p > best) {
      best = p;
      bestB1 = B1;
      bestB2 = B2;
      bestR = r;
    }
  }
  printf("\n%u %u %u %f\n", bestR, bestB1, bestB2, best * 100);  
}

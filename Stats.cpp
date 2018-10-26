// Copyright Mihai Preda

#include "Stats.h"

#include <cmath>
#include <cassert>
#include <numeric>

using namespace std;

void Stats::reset() {
  times.clear();
  nSquares.clear();
  nMuls.clear();
}

void Stats::add(double millis, u32 nSq, u32 nMul) {
  times.push_back(millis);
  nSquares.push_back(nSq);
  nMuls.push_back(nMul);  
}

template<typename T1, typename T2> static auto dotProduct(const vector<T1> &v1, const vector<T2> &v2) {
  return inner_product(v1.begin(), v1.end(), v2.begin(), 0);
}

template<typename T> static T sum(const vector<T> &v) { return accumulate(v.begin(), v.end(), 0); }

StatsInfo Stats::getStats() {
  u32 totalSq  = sum(nSquares);
  u32 totalMul = sum(nMuls);
  double totalTime = sum(times);
  
  if (totalMul == 0) { return StatsInfo{totalSq, 0u, totalTime / totalSq, 0.0}; }
  
  double a = dotProduct(nSquares, nSquares);
  double b = dotProduct(nSquares, nMuls);
  double d = dotProduct(nMuls, nMuls);
  double e = dotProduct(nSquares, times);
  double f = dotProduct(nMuls, times);
  
  assert(a && d);

  double det = a * d - b * b;
  double x = d * e - b * f;
  double y = -b * e + a * f;
  return {totalSq, totalMul, x / det, y / det};
}

#include "Stats.h"

#include <cmath>
#include <cassert>

void Stats::reset() {
  v.clear();
  min =   1e100;
  max = - 1e100;
  sum = 0;
}
  
void Stats::add(double x) {
  min = std::min(x, min);
  max = std::max(x, max);
  sum += x;
  v.push_back(x);
}

StatsInfo Stats::getStats() {
  int n = v.size();
  double mean = n ? sum / n : 0;
  
  int n1 = 0, n2 = 0;
  double s1 = 0, s2 = 0;
  
  for (double x : v) {
    double d = x - mean;
    double dd = d * d;
    if (d < 0) {
      ++n1;
      s1 += dd;
    } else {
      ++n2;
      s2 += dd;
    }
  }
  assert(n == n1 + n2);

  double a = n1 ? mean - sqrt(s1 / n1) : mean;
  double b = n2 ? mean + sqrt(s2 / n2) : mean;
  // (s2 * n2 / (n1 * n));
  // (s2 * n1 / (n2 * n));
  return StatsInfo{n, mean, min, max, a, b, sum};
}

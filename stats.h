#pragma once

#include <vector>
#include <algorithm>

#include <cmath>
#include <cassert>

struct StatsInfo {
  int n;
  double mean, min, max, low, high, sum;
};

class Stats {  
  std::vector<double> v;
  double min, max, sum;

public:  
  Stats() { reset(); }
  
  void reset() {
    v.clear();
    min =   1e100;
    max = - 1e100;
    sum = 0;
  }
  
  void add(double x) {
    min = std::min(x, min);
    max = std::max(x, max);
    sum += x;
    v.push_back(x);
  }

  StatsInfo getStats() {
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
};

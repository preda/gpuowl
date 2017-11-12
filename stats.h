#include <algorithm>
#include <cmath>

struct Stats {
  int n;
  double min, max;
  double mean, m2;

  Stats() { reset(); }
  
  void reset() {
    n = 0;
    min =   1e100;
    max = - 1e100;
    mean = 0;
    m2 = 0;
  }
  
  void add(double x) {
    min = std::min(x, min);
    max = std::max(x, max);
    
    ++n;
    double delta = x - mean;
    mean += delta / n;
    double delta2 = x - mean;
    m2 += delta * delta2;
  }

  double variance() { return m2 / (n - 1); }
  double sd() { return sqrt(variance()); }
};

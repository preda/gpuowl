#pragma once

#include <vector>

struct StatsInfo {
  int n;
  double mean, min, max, low, high, sum;
};

class Stats {  
  std::vector<double> v;
  double min, max, sum;

public:  
  Stats() { reset(); }
  void reset();
  void add(double x);
  StatsInfo getStats();
};

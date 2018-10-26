// Copyright Mihai Preda.

#pragma once

#include "common.h"

#include <vector>

struct StatsInfo {
  u32 nSq;
  u32 nMul;
  double msPerSq;
  double msPerMul;
};

class Stats {
  vector<double> times;
  vector<u32> nSquares;
  vector<u32> nMuls;


public:  
  Stats() { reset(); }
  
  void reset();
  void add(double millis, u32 nSq, u32 nMul);
  StatsInfo getStats();
};

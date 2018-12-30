// Copyright Mihai Preda.

#pragma once

#include "common.h"

#include <vector>

struct StatsInfo {
  u32 nSq;
  double msPerSq;
  double msPerIt;
  // double msPerMul;
};

class Stats {
  double time;
  u32 nSq;

public:
  Stats() { reset(); }
  void add(double millis, u32 nSq);
  StatsInfo reset();
};

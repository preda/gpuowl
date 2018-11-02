// Copyright Mihai Preda.

#pragma once

#include "common.h"

#include <vector>

struct StatsInfo {
  u32 nSq;
  u32 nMul;
  double msPerSq;
  double msPerIt;
  // double msPerMul;
};

class Stats {
  double time;
  u32 nSq;
  u32 nMul;

public:
  Stats() { reset(); }
  void add(double millis, u32 nSq, u32 nMul);
  StatsInfo reset();
};

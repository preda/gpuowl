// Copyright Mihai Preda

#include "Stats.h"

#include <cmath>
#include <cassert>
#include <numeric>

void Stats::add(double millis, u32 sq, u32 mul) {
  time += millis;
  nSq  += sq;
  nMul += mul;
}

StatsInfo Stats::reset() {
  StatsInfo ret{nSq, nMul, time / (nSq + nMul * 1.19), time / nSq};
  nSq = nMul = 0;
  time = 0;
  return ret;
}

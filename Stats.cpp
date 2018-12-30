// Copyright Mihai Preda

#include "Stats.h"

#include <cmath>
#include <cassert>
#include <numeric>

void Stats::add(double millis, u32 sq) {
  time += millis;
  nSq  += sq;
}

StatsInfo Stats::reset() {
  StatsInfo ret{nSq, time / nSq};
  nSq = 0;
  time = 0;
  return ret;
}

// Copyright Mihai Preda.

#pragma once

#include "common.h"

struct Stats {
  double time = 0;
  u32 nSq = 0;

  void add(double millis, u32 deltaN) { time += millis; nSq += deltaN; }
  void reset() { time = 0; nSq = 0; }
};

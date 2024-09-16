// Copyright Mihai Preda

#pragma once

#include "common.h"
#include <array>

struct TrigCoefs {
  int scale;
  std::array<double, 7> sinCoefs;
  std::array<double, 7> cosCoefs;
};

TrigCoefs trigCoefs(u32 N);

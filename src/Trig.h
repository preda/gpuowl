// Copyright Mihai Preda

#pragma once

#include "common.h"
#include <array>

struct TrigCoefs {
  u32 scaleSin;
  u32 scaleCos;
  std::array<double, 8> sinCoefs;
  std::array<double, 8> cosCoefs;
};

TrigCoefs trigCoefs(u32 N);

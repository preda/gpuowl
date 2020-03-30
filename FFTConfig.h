// Copyright Mihai Preda

#pragma once

#include "common.h"

#include <string>
#include <vector>

// Format 'n' with a K or M suffix if multiple of 1024 or 1024*1024
string numberK(u32 n);

struct FFTConfig {
  u32 width  = 0;
  u32 height = 0;
  u32 middle = 0;
  u32 fftSize = 0;
  u32 maxExp = 0;

  static u32 getMaxExp(u32 fftSize, bool isPm1);
  static u32 getMaxCarry32(u32 fftSize, u32 exponent);

  static std::vector<FFTConfig> genConfigs(bool isPm1);

  static std::string configName(u32 width, u32 height, u32 middle);
  
  FFTConfig(u32 width, u32 height, u32 middle, bool isPm1);

  FFTConfig() {}
};

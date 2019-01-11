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

  static u32 getMaxExp(u32 fftSize);

  static std::vector<FFTConfig> genConfigs();

  static std::string configName(u32 width, u32 height, u32 middle);
  
  FFTConfig(u32 width, u32 height, u32 middle);

  FFTConfig() {}
};

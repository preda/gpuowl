// Copyright Mihai Preda

#pragma once

#include "common.h"

#include <string>
#include <tuple>
#include <vector>
#include <cmath>

// Format 'n' with a K or M suffix if multiple of 1024 or 1024*1024
string numberK(u32 n);

struct FFTConfig {
  static constexpr const float MIN_BPW = 2.5;

  static u32 getMaxExp(u32 fftSize, u32 middle) {
    return 6 * fftSize;
 }
  
  static std::vector<FFTConfig> genConfigs();

  // FFTConfig(u32 w, u32 m, u32 h) : width(w), middle(m), height(h) {}
  static FFTConfig fromSpec(const string& spec);
  
  u32 width  = 0;
  u32 middle = 0;
  u32 height = 0;
    
  u32 fftSize() const { return width * height * middle * 2; }
  u32 maxExp() const { return getMaxExp(fftSize(), middle); }
  std::string spec() const { return numberK(width) + ':' + numberK(middle) + ':' + numberK(height); }
};

// Copyright (C) Mihai Preda and George Woltman

#pragma once

#include "common.h"

#include <string>
#include <tuple>
#include <vector>
#include <cmath>

// Format 'n' with a K or M suffix if multiple of 1024 or 1024*1024
string numberK(u32 n);

using KeyVal = std::pair<std::string, std::string>;

struct FFT {
  u32 priority;
  u32 maxExp;
  std::string fft;
  std::vector<KeyVal> uses;
};

class FFTShape {
  vector<FFT> readTune();

public:
  static constexpr const float MIN_BPW = 3;

  static u32 getMaxExp(u32 fftSize, u32 middle) { return
      middle == 2 ? fftSize * (19.0766 - 0.279 * log2(fftSize / (1.0 * 1024 * 1024))) :
                middle == 3 ? fftSize * (19.0766 - 0.279 * log2(fftSize / (1.5 * 1024 * 1024))) :
                middle == 4 ? fftSize * (18.9862 - 0.279 * log2(fftSize / (2.0 * 1024 * 1024))) :
                middle == 5 ? fftSize * (18.8482 - 0.279 * log2(fftSize / (2.5 * 1024 * 1024))) :
                middle == 6 ? fftSize * (18.7810 - 0.279 * log2(fftSize / (3.0 * 1024 * 1024))) :
                middle == 7 ? fftSize * (18.7113 - 0.279 * log2(fftSize / (3.5 * 1024 * 1024))) :
                middle == 8 ? fftSize * (18.6593 - 0.279 * log2(fftSize / (4.0 * 1024 * 1024))) :
                middle == 9 ? fftSize * (18.6135 - 0.279 * log2(fftSize / (4.5 * 1024 * 1024))) :
                middle == 10 ? fftSize * (18.5719 - 0.279 * log2(fftSize / (5.0 * 1024 * 1024))) :
                middle == 11 ? fftSize * (18.5317 - 0.279 * log2(fftSize / (5.5 * 1024 * 1024))) :
                middle == 12 ? fftSize * (18.5185 - 0.279 * log2(fftSize / (6.0 * 1024 * 1024))) :
                middle == 13 ? fftSize * (18.4795 - 0.279 * log2(fftSize / (6.5 * 1024 * 1024))) :
                middle == 14 ? fftSize * (18.4451 - 0.279 * log2(fftSize / (7.0 * 1024 * 1024))) :
			       fftSize * (18.3804 - 0.279 * log2(fftSize / (7.5 * 1024 * 1024))); }
  
  static u32 getMaxCarry32(u32 fftSize, u32 exponent);
  static std::vector<FFTShape> genConfigs();

  static tuple<u32, u32, bool> getChainLengths(u32 fftSize, u32 exponent, u32 middle);

  // FFTShape(u32 w, u32 m, u32 h) : width(w), middle(m), height(h) {}
  static FFTShape fromSpec(const string& spec);
  static vector<FFTShape> multiSpec(const string& spec);
  
  u32 width  = 0;
  u32 middle = 0;
  u32 height = 0;
    
  u32 fftSize() const { return width * height * middle * 2; }
  u32 maxExp() const { return getMaxExp(fftSize(), middle); }
  std::string spec() const { return numberK(width) + ':' + numberK(middle) + ':' + numberK(height); }
};

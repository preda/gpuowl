// Copyright (C) Mihai Preda.

#include "FFTConfig.h"
#include "common.h"

#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;

FFTConfig::FFTConfig(u32 width, u32 height, u32 middle, bool isPm1) :
  width(width),
  height(height),
  middle(middle),
  fftSize(width * height * middle * 2),

  maxExp(getMaxExp(fftSize, isPm1)) {
  assert(width == 256 || width  == 512 || width == 1024 || width == 2048 || width == 4096);
  assert(height == 256 || height == 512 || height == 1024 || height == 2048);
}

u32 FFTConfig::getMaxExp(u32 fftSize, bool isPm1) {
  return fftSize * ((isPm1 ? 18.0 : 18.257) - 0.33 * log2(fftSize * (1.0 / (9 * 512 * 1024))));
}
// { return fftSize * (17.77 + 0.33 * (24 - log2(fftSize))); }
// 17.88 + 0.36 * (24 - log2(n)); Update after feedback on 86700001, FFT 4608 (18.37b/w) being insufficient.

vector<FFTConfig> FFTConfig::genConfigs(bool isPm1) {
  vector<FFTConfig> configs;
  for (u32 width : {256, 512, 1024, 2048, 4096}) {
    for (u32 height : {256, 512, 1024, 2048}) {
      for (u32 middle : {1, /*3,*/ 4, /*5,*/ 6, 7, 8, 9, 10, 11, 12}) {
        configs.push_back(FFTConfig(width, height, middle, isPm1));
      }
    }
  }
  std::sort(configs.begin(), configs.end(), [](const FFTConfig &a, const FFTConfig &b) {
      if (a.fftSize != b.fftSize) { return (a.fftSize < b.fftSize); }

      if (a.width != b.width) {
        if (a.width == 1024 || b.width == 1024) { return a.width == 1024; }
        return a.width < b.width;
      }

      return a.height < b.height;
    });
  return configs;
}

string numberK(u32 n) {
  return (n % (1024 * 1024) == 0) ? to_string(n / (1024 * 1024)) + "M" : (n % 1024 == 0) ? to_string(n / 1024) + "K" : to_string(n);
}

string FFTConfig::configName(u32 width, u32 height, u32 middle) {
  return numberK(width) + '-' + numberK(height) + ((middle != 1) ? "-"s + numberK(middle) : ""s);
}

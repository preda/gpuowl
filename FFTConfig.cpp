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
  assert(width == 256 || width  == 512 || width == 1024 || width == 4096);
  assert(height == 256 || height == 512 || height == 1024);
}

// On 2020-03-30, I examined the middle=10 FFTs from 1.25M to 80M.
// On this date, exponent 95460001 had an average roundoff error of 0.2441.
// This should be periodically tested to make sure rocm optimizer hasn't made accuracy worse.
//
// I'm targetting an average max roundoff of 0.262, which ought to give us some roundoff
// errors above 0.4 and I hope none above 0.5.  The 1.25M FFT ended up with 18.814 bits-per-word
// and the 80M FFT ended up with 17.141 bits-per-word.  This gives a simple formula of
//		bits-per-word = 18.814 - 0.279 * log2 (FFTsize / 1.25M)
// At a later date, we might should create a different formula for each Middle value as
// the multiplication chains in MiddleIn/Out may have a big affect on the roundoff error.
//
// Also note, that I did not see any evidence that we need to be more conservative during P-1.
// However, P-1 does not output average max roundoff error, so I'm not 100% confident.

u32 FFTConfig::getMaxExp(u32 fftSize, bool isPm1) {
  return fftSize * (18.814 - 0.279 * log2(fftSize / (1.25 * 1024 * 1024)));
}

// This routine predicts the maximum carry32 we might see.  This was based on 500,000 iterations
// of 24518003 using a 1.25M FFT.  The maximum carry32 value observed was 0x32420000.
// As FFT length grows, so does the expected max carry32.  As we store fewer bits-per-word in
// an FFT size, the expected max carry32 decreases.  Our formula is:
//		max carry32 = 0x32420000 * 2^(BPW - 18.706) * 2 ^ (2 * 0.279 * log2(FFTSize / 1.25M))
//
// Note that the mul-by-3 carryFusedMul kernel, triples the expected max carry32 value.
// As of now, I have limited data on the carryFusedMulDelta kernel.
//
// Note: This routine returns the top 16 bits of the expected max carry32.

u32 FFTConfig::getMaxCarry32(u32 fftSize, u32 exponent) {
  return (u32) (0x3242 * pow (2.0, 0.558 * log2(fftSize / (1.25 * 1024 * 1024)) + double(exponent) / double(fftSize) - 18.706));
}

vector<FFTConfig> FFTConfig::genConfigs(bool isPm1) {
  vector<FFTConfig> configs;
  for (u32 width : {256, 512, 1024, 4096}) {
    for (u32 height : {256, 512, 1024}) {
      for (u32 middle : {1, /*3,*/ 4, /*5,*/ 6, 7, 8, 9, 10, 11, 12}) {
        if (middle == 1 && width * height >= 512 * 512) continue;
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

// Copyright (C) Mihai Preda and George Woltman

#pragma once

#include "common.h"

#include <string>
#include <tuple>
#include <vector>
#include <array>
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
  array<double, 4> bpw;

public:
  static constexpr const float MIN_BPW = 3;
  
  static u32 getMaxCarry32(u32 fftSize, u32 exponent);
  static std::vector<FFTShape> genConfigs();

  static tuple<u32, u32, bool> getChainLengths(u32 fftSize, u32 exponent, u32 middle);

  FFTShape(u32 w, u32 m, u32 h);
  // : width(w), middle(m), height(h) {}
  static FFTShape fromSpec(const string& spec);
  static vector<FFTShape> multiSpec(const string& spec);
  
  u32 width  = 0;
  u32 middle = 0;
  u32 height = 0;

    
  u32 fftSize() const { return width * height * middle * 2; }
  double maxBpw() const { return bpw[3]; }
  u32 maxExp()  const { return maxBpw() * fftSize(); }
  std::string spec() const { return numberK(width) + ':' + numberK(middle) + ':' + numberK(height); }
};

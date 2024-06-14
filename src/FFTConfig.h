// Copyright (C) Mihai Preda and George Woltman

#pragma once

#include "Args.h"
#include "common.h"

#include <string>
#include <tuple>
#include <vector>
#include <array>
#include <algorithm>

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
  
  static u32 getMaxCarry32(u32 fftSize, u32 exponent);
  static std::vector<FFTShape> genConfigs();

  static tuple<u32, u32, bool> getChainLengths(u32 fftSize, u32 exponent, u32 middle);

  static FFTShape fromSpec(const string& spec);
  static vector<FFTShape> multiSpec(const string& spec);
  
  u32 width  = 0;
  u32 middle = 0;
  u32 height = 0;
  array<double, 4> bpw;

  FFTShape(u32 w, u32 m, u32 h);

  u32 fftSize() const { return width * height * middle * 2; }
  double maxBpw() const { return *max_element(bpw.begin(), bpw.end()); }
  std::string spec() const { return numberK(width) + ':' + numberK(middle) + ':' + numberK(height); }
};

struct FFTConfig {
public:
  static const u32 N_VARIANT = 4;

  /*
  std::string variantSpec() const { return variantSpec(variant); }

  enum Variant {
    CLEAN0_TRIG0 = 0,
    CLEAN0_TRIG1,
    CLEAN1_TRIG0,
    CLEAN1_TRIG1,
  };

  static std::string variantSpec(Variant variant) {
    switch (variant) {
      case CLEAN0_TRIG0: return "CLEAN=0,TRIG_HI=0";
      case CLEAN0_TRIG1: return "CLEAN=0,TRIG_HI=1";
      case CLEAN1_TRIG0: return "CLEAN=1,TRIG_HI=0";
      case CLEAN1_TRIG1: return "CLEAN=1,TRIG_HI=1";
    }
  }
  */

  FFTShape shape;
  u32 variant;

  double maxBpw() const { return shape.bpw[variant]; }
  u32 maxExp()  const { return maxBpw() * shape.fftSize(); }
  std::string spec() const { return shape.spec() + ":" + to_string(variant); }

  static FFTConfig bestFit(u32 E, const std::string& spec);
};




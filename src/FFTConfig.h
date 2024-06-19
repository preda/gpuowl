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

class FFTShape {
public:
  static constexpr const float MIN_BPW = 3;
  
  u32 getMaxCarry32(u32 exponent) const;
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
  u32 nW() const { return (width == 1024 || width == 256 || width == 4096) ? 4 : 8; }
  u32 nH() const { return (height == 1024 || height == 256 || height == 4096) ? 4 : 8; }

  double maxBpw() const { return *max_element(bpw.begin(), bpw.end()); }
  std::string spec() const { return numberK(width) + ':' + numberK(middle) + ':' + numberK(height); }
};

struct FFTConfig {
  bool matches(const string& spec) const;

public:
  static const u32 N_VARIANT = 4;

  static FFTConfig bestFit(u32 E, const std::string& spec);

  FFTShape shape;
  u32 variant;
  std::vector<KeyVal> config{};

  explicit FFTConfig(const string& spec);
  FFTConfig(FFTShape shape, u32 variant) : shape{shape}, variant{variant} {}

  double maxBpw() const { return shape.bpw[variant]; }
  u32 maxExp()  const { return maxBpw() * shape.fftSize(); }
  std::string spec() const { return shape.spec() + ":" + to_string(variant); }
};

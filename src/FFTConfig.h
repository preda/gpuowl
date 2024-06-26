// Copyright (C) Mihai Preda and George Woltman

#pragma once

#include "common.h"

#include <string>
#include <tuple>
#include <vector>
#include <array>
#include <algorithm>

class Args;

// Format 'n' with a K or M suffix if multiple of 1024 or 1024*1024
string numberK(u32 n);

using KeyVal = std::pair<std::string, std::string>;

class FFTShape {
public:
  static constexpr const float MIN_BPW = 3;
  
  static std::vector<FFTShape> allShapes(u32 from=0, u32 to = -1);

  static tuple<u32, u32, bool> getChainLengths(u32 fftSize, u32 exponent, u32 middle);

  static vector<FFTShape> multiSpec(const string& spec);
  
  u32 width  = 0;
  u32 middle = 0;
  u32 height = 0;
  array<double, 4> bpw;

  FFTShape(u32 w = 1, u32 m = 1, u32 h = 1);
  FFTShape(const string& w, const string& m, const string& h);
  explicit FFTShape(const string& spec);

  u32 size() const { return width * height * middle * 2; }
  u32 nW() const { return (width == 1024 || width == 256 /*|| width == 4096*/) ? 4 : 8; }
  u32 nH() const { return (height == 1024 || height == 256 /*|| height == 4096*/) ? 4 : 8; }

  double maxBpw() const { return *max_element(bpw.begin(), bpw.end()); }
  std::string spec() const { return numberK(width) + ':' + numberK(middle) + ':' + numberK(height); }

  double carry32BPW() const;
  bool needsLargeCarry(u32 E) const;
};

struct FFTConfig {
public:
  static const u32 N_VARIANT = 4;
  static FFTConfig bestFit(const Args& args, u32 E, const std::string& spec);

  enum CARRY_KIND { CARRY_AUTO, CARRY_32, CARRY_64};

  FFTShape shape{};
  u32 variant;
  u32 carry;

  explicit FFTConfig(const string& spec);
  FFTConfig(FFTShape shape, u32 variant, u32 carry /* = CARRY_AUTO*/);

  std::string spec() const;
  u32 size() const { return shape.size(); }
  u32 maxExp()  const { return maxBpw() * shape.size(); }

  double maxBpw() const;

  // bool needsLargeCarry(u32 E) const { return shape.needsLargeCarry(E); }
};

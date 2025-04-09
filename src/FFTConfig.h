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

static const u32 N_VARIANT_W = 4;
static const u32 N_VARIANT_M = 2;
static const u32 N_VARIANT_H = 4;
static const u32 LAST_VARIANT = (N_VARIANT_W - 1) * 100 + (N_VARIANT_M - 1) * 10 + N_VARIANT_H - 1;
inline u32 variant_WMH(u32 v_W, u32 v_M, u32 v_H) { return v_W * 100 + v_M * 10 + v_H; }
inline u32 variant_W(u32 v) { return v / 100; }
inline u32 variant_M(u32 v) { return v % 100 / 10; }
inline u32 variant_H(u32 v) { return v % 10; }
inline u32 next_variant(u32 v) { u32 new_v;
  new_v = v + 1; if (variant_H (new_v) < N_VARIANT_H) return (new_v);
  new_v = (v / 10 + 1) * 10; if (variant_M (new_v) < N_VARIANT_M) return (new_v);
  new_v = (v / 100 + 1) * 100; return (new_v);
}

enum CARRY_KIND { CARRY_32=0, CARRY_64=1, CARRY_AUTO=2};

struct FFTConfig {
public:
  static FFTConfig bestFit(const Args& args, u32 E, const std::string& spec);

  FFTShape shape{};
  u32 variant;
  u32 carry;

  explicit FFTConfig(const string& spec);
  FFTConfig(FFTShape shape, u32 variant, u32 carry);

  std::string spec() const;
  u32 size() const { return shape.size(); }
  u32 maxExp()  const { return maxBpw() * shape.size(); }

  double maxBpw() const;
};

// Copyright (C) Mihai Preda.

#include "FFTConfig.h"
#include "Args.h"
#include "common.h"
#include "log.h"
#include "TuneEntry.h"

#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>
#include <array>
#include <map>
#include <cinttypes>

using namespace std;

struct FftBpw {
  string fft;
  array<float, NUM_BPW_ENTRIES> bpw;
};

map<string, array<float, NUM_BPW_ENTRIES>> BPW {
#include "fftbpw.h"
};

namespace {

u32 parseInt(const string& s) {
  // if (s.empty()) { return 1; }
  assert(!s.empty());
  char c = s.back();
  u32 multiple = c == 'k' || c == 'K' ? 1024 : c == 'm' || c == 'M' ? 1024 * 1024 : 1;
  return strtod(s.c_str(), nullptr) * multiple;
}

} // namespace

// Accepts:
// - a prefix indicating FFT_type (if not specified, default is FP64)
// - a single config: 1K:13:256
// - a size: 6.5M
// - a range of sizes: 6.5M-7M
// - a list: 6M-7M,1K:13:256
vector<FFTShape> FFTShape::multiSpec(const string& iniSpec) {
  if (iniSpec.empty()) { return allShapes(); }

  vector<FFTShape> ret;

  for (const string &spec : split(iniSpec, ',')) {
    enum FFT_TYPES fft_type = FFT64;
    auto parts = split(spec, ':');
    if (parseInt(parts[0]) < 60) {      // Look for a prefix specifying the FFT type
      fft_type = (enum FFT_TYPES) parseInt(parts[0]);
      parts = vector(next(parts.begin()), parts.end());
    }
    assert(parts.size() <= 3);
    if (parts.size() == 3) {
      u32 width = parseInt(parts[0]);
      u32 middle = parseInt(parts[1]);
      u32 height = parseInt(parts[2]);
      ret.push_back({fft_type, width, middle, height});
      continue;
    }
    assert(parts.size() == 1);

    parts = split(spec, '-');
    assert(parts.size() >= 1 && parts.size() <= 2);
    u32 sizeFrom = parseInt(parts[0]);
    u32 sizeTo = parts.size() == 2 ? parseInt(parts[1]) : sizeFrom;
    auto shapes = allShapes(sizeFrom, sizeTo);
    if (shapes.empty()) {
      log("Could not find a FFT config for '%s'\n", spec.c_str());
      throw "Invalid FFT spec";
    }
    ret.insert(ret.end(), shapes.begin(), shapes.end());
  }
  return ret;
}

vector<FFTShape> FFTShape::allShapes(u32 sizeFrom, u32 sizeTo) {
  vector<FFTShape> configs;
  for (enum FFT_TYPES type : {FFT64, FFT3161, FFT3261, FFT61, FFT323161}) {
    for (u32 width : {256, 512, 1024, 4096}) {
      for (u32 height : {256, 512, 1024}) {
        if (width == 256 && height == 1024) { continue; } // Skip because we prefer width >= height
        for (u32 middle : {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}) {
          if (type != FFT64 && (middle & (middle - 1))) continue;   // Reject non-power-of-two NTTs
          u32 sz = width * height * middle * 2;
          if (sizeFrom <= sz && sz <= sizeTo) {
            configs.push_back({type, width, middle, height});
          }
        }
      }
    }
  }
  std::sort(configs.begin(), configs.end(),
            [](const FFTShape &a, const FFTShape &b) {
              if (a.size() != b.size()) { return (a.size() < b.size()); }
              if (a.width != b.width) {
                if (a.width == 1024 || b.width == 1024) { return a.width == 1024; }
                return a.width < b.width;
              }
              return a.height < b.height;
            });
  return configs;
}

FFTShape::FFTShape(const string& spec) {
  assert(!spec.empty());
  enum FFT_TYPES fft_type = FFT64;
  vector<string> v = split(spec, ':');
  if (parseInt(v[0]) < 60) {      // Look for a prefix specifying the FFT type
    fft_type = (enum FFT_TYPES) parseInt(v[0]);
    for (u32 i = 1; i < v.size(); ++i) v[i-1] = v[i];
    v.resize(v.size() - 1);
  }
  assert(v.size() == 3);
  *this = FFTShape{fft_type, v.at(0), v.at(1), v.at(2)};
}

FFTShape::FFTShape(enum FFT_TYPES t, const string& w, const string& m, const string& h) :
  FFTShape{t, parseInt(w), parseInt(m), parseInt(h)}
{}

FFTShape::FFTShape(u32 w, u32 m, u32 h) :
  FFTShape(FFT64, w, m, h) {
}

FFTShape::FFTShape(enum FFT_TYPES t, u32 w, u32 m, u32 h) :
  fft_type{t}, width{w}, middle{m}, height{h} {
  assert(w && m && h);

  // Un-initialized shape, don't set BPW
  if (w == 1 && m == 1 && h == 1) { return; }

  string s = spec();
  if (auto it = BPW.find(s); it != BPW.end()) {
    bpw = it->second;
  } else {
    if (height > width) {
      bpw = FFTShape{h, m, w}.bpw;
    } else {
      // Make up some defaults

      //double d = 0.275 * (log2(size()) - log2(256 * 13 * 1024 * 2));
      //bpw = {18.1-d, 18.2-d, 18.2-d, 18.3-d};
      //log("BPW info for %s not found, defaults={%.2f, %.2f, %.2f, %.2f}\n", s.c_str(), bpw[0], bpw[1], bpw[2], bpw[3]);

      // Manipulate the shape into something that was likely pre-computed
      while (m < 9) { m *= 2; w /= 2; }
      while (w >= 4*h) { w /= 2; h *= 2; }
      while (w < h || w < 256 || w == 2048) { w *= 2; h /= 2; }
      while (h < 256) { h *= 2; m /= 2; }
      if (m == 1) m = 2;
      bpw = FFTShape{w, m, h}.bpw;
      for (u32 j = 0; j < NUM_BPW_ENTRIES; ++j) bpw[j] -= 0.05;   // Assume this fft spec is worse than measured fft specs
      if (this->isFavoredShape()) {  // Don't output this warning message for non-favored shapes (we expect the BPW info to be missing)
        printf("BPW info for %s not found, defaults={", s.c_str());
        for (u32 j = 0; j < NUM_BPW_ENTRIES; ++j) printf("%s%.2f", j ? ", " : "", (double) bpw[j]);
        printf("}\n");
      }
    }
  }
}

float FFTShape::carry32BPW() const {
  // The formula below was validated empirically with -carryTune

  // We observe that FFT 6.5M (1024:13:256) has safe carry32 up to 18.35 BPW
  // while the 0.5*log2() models the impact of FFT size changes.
  // We model carry with a Gumbel distrib similar to the one used for ROE, and measure carry with
  // -use STATS=1. See -carryTune

//GW:  I have no idea why this is needed.  Without it, -tune fails on FFT sizes from 256K to 1M
// Perhaps it has something to do with RNDVALdoubleToLong in carryutil
if (18.35 + 0.5 * (log2(13 * 1024 * 512) - log2(size())) > 19.0) return 19.0;

  return 18.35 + 0.5 * (log2(13 * 1024 * 512) - log2(size()));
}

bool FFTShape::needsLargeCarry(u32 E) const {
  return E / double(size()) > carry32BPW();
}

// Return TRUE for "favored" shapes.  That is, those that are most likely to be useful.  To save time in generating bpw data, only these favored
// shapes have their bpw data pre-computed.  Bpw for non-favored shapes is guessed from the bpw data we do have.  Also. -tune will normally only
// time favored shapes.  These are the rules for deciding favored shapes:
//      WIDTH >= HEIGHT
//      WIDTH=4K:  HEIGHT>=512, MIDDLE>=9       (2*8 combos)
//      WIDTH=1K:  MIDDLE>=5                    (3*12 combos)
//      WIDTH=512: MIDDLE>=4                    (2*13 combos)
//      WIDTH=256: MIDDLE>=1                    (16 combos)
bool FFTShape::isFavoredShape() const {
  return width >= height &&
        ((width == 4096 && height >= 512 && middle >= 9) ||
         (width == 1024 && middle >= 5) ||
         (width == 512 && middle >= 4) ||
         (width == 256 && middle >= 1));
}

FFTConfig::FFTConfig(const string& spec) {
  auto v = split(spec, ':');

  enum FFT_TYPES fft_type = FFT64;
  if (parseInt(v[0]) < 60) {      // Look for a prefix specifying the FFT type
    fft_type = (enum FFT_TYPES) parseInt(v[0]);
    for (u32 i = 1; i < v.size(); ++i) v[i-1] = v[i];
    v.resize(v.size() - 1);
  }

  if (v.size() == 1) {
    *this = {FFTShape::multiSpec(spec).front(), LAST_VARIANT, CARRY_AUTO};
  } if (v.size() == 3) {
    *this = {FFTShape{fft_type, v[0], v[1], v[2]}, LAST_VARIANT, CARRY_AUTO};
  } else if (v.size() == 4) {
    *this = {FFTShape{fft_type, v[0], v[1], v[2]}, parseInt(v[3]), CARRY_AUTO};
  } else if (v.size() == 5) {
    int c = parseInt(v[4]);
    assert(c == 0 || c == 1);
    *this = {FFTShape{fft_type, v[0], v[1], v[2]}, parseInt(v[3]), c == 0 ? CARRY_32 : CARRY_64};
  } else {
    throw "FFT spec";
  }
}

FFTConfig::FFTConfig(FFTShape shape, u32 variant, u32 carry) :
  shape{shape},
  variant{variant},
  carry{carry}
{
  assert(variant_W(variant) < N_VARIANT_W);
  assert(variant_M(variant) < N_VARIANT_M);
  assert(variant_H(variant) < N_VARIANT_H);

  if      (shape.fft_type == FFT64)     FFT_FP64 = 1, FFT_FP32 = 0, NTT_GF31 = 0, NTT_GF61 = 0, WordSize = 4;
  else if (shape.fft_type == FFT3161)   FFT_FP64 = 0, FFT_FP32 = 0, NTT_GF31 = 1, NTT_GF61 = 1, WordSize = 8;
  else if (shape.fft_type == FFT3261)   FFT_FP64 = 0, FFT_FP32 = 1, NTT_GF31 = 0, NTT_GF61 = 1, WordSize = 8;
  else if (shape.fft_type == FFT61)     FFT_FP64 = 0, FFT_FP32 = 0, NTT_GF31 = 0, NTT_GF61 = 1, WordSize = 4;
  else if (shape.fft_type == FFT323161) FFT_FP64 = 0, FFT_FP32 = 1, NTT_GF31 = 1, NTT_GF61 = 1, WordSize = 8;
  else if (shape.fft_type == FFT3231)   FFT_FP64 = 0, FFT_FP32 = 1, NTT_GF31 = 1, NTT_GF61 = 0, WordSize = 4;
  else if (shape.fft_type == FFT6431)   FFT_FP64 = 1, FFT_FP32 = 0, NTT_GF31 = 1, NTT_GF61 = 0, WordSize = 8;
  else if (shape.fft_type == FFT31)     FFT_FP64 = 0, FFT_FP32 = 0, NTT_GF31 = 1, NTT_GF61 = 0, WordSize = 4;
  else if (shape.fft_type == FFT32)     FFT_FP64 = 0, FFT_FP32 = 1, NTT_GF31 = 0, NTT_GF61 = 0, WordSize = 4;
  else throw "FFT type";
}

string FFTConfig::spec() const {
  string s = shape.spec() + ":" + to_string(variant_W(variant)) + to_string(variant_M(variant)) + to_string(variant_H(variant));
  return carry == CARRY_AUTO ? s : (s + (carry == CARRY_32 ? ":0" : ":1"));
}

float FFTConfig::maxBpw() const {
  float b;
  // Look up the pre-computed maximum bpw.  The lookup table contains data for variants 000, 101, 202, 010, 111, 212.
  // For 4K width, the lookup table contains data for variants 100, 101, 202, 110, 111, 212 since BCAST only works for width <= 1024.
  if (variant_W(variant) == variant_H(variant) ||
      (shape.width > 1024 && variant_W(variant) == 1 && variant_H(variant) == 0)) {
    b = shape.bpw[variant_M(variant) * 3 + variant_H(variant)];
  }
  // Interpolate for the maximum bpw.  This might could be improved upon.  However, I doubt people will use these variants often.
  else {
    float b1 = shape.bpw[variant_M(variant) * 3 + variant_W(variant)];
    float b2 = shape.bpw[variant_M(variant) * 3 + variant_H(variant)];
    b = (b1 + b2) / 2.0;
  }
  // Only some FFTs support both 32 and 64 bit carries.
  return (carry == CARRY_32 && (shape.fft_type == FFT64 || shape.fft_type == FFT3231)) ? std::min(shape.carry32BPW(), b) : b;
}

FFTConfig FFTConfig::bestFit(const Args& args, u32 E, const string& spec) {
  // A FFT-spec was given, simply take the first FFT from the spec that can handle E
  if (!spec.empty()) {
    FFTConfig fft{spec};
    if (fft.maxExp() * args.fftOverdrive < E) {
      log("Warning: %s (max %" PRIu64 ") may be too small for %u\n", fft.spec().c_str(), fft.maxExp(), E);
    }
    return fft;
  }

  // No FFT-spec given, so choose from tune.txt the fastest FFT that can handle E
  vector<TuneEntry> tunes = TuneEntry::readTuneFile(args);
  for (const TuneEntry& e : tunes) {
    // The first acceptable is the best as they're sorted by cost
    if (E <= e.fft.maxExp() * args.fftOverdrive) { return e.fft; }
  }

  log("No FFTs found in tune.txt that can handle %u. Consider tuning with -tune\n", E);

  // Take the first FFT that can handle E
  for (const FFTShape& shape : FFTShape::allShapes()) {
    for (u32 v : {101, 202}) {
      if (FFTConfig fft{shape, v, CARRY_AUTO}; fft.maxExp() * args.fftOverdrive >= E) { return fft; }
    }
  }

  log("No FFT found for %u\n", E);
  throw "No FFT";
}


string numberK(u64 n) {
  u32 K = 1024;
  u32 M = K * K;

  if (n % M == 0) { return to_string(n / M) + 'M'; }

  char buf[64];
  if (n >= M && (n * u64(100)) % M == 0) {
    snprintf(buf, sizeof(buf), "%.2f", float(n) / M);
    return string(buf) + 'M';
  } else if (n >= K) {
    snprintf(buf, sizeof(buf), "%g", float(n) / K);
    return string(buf) + 'K';
  } else {
    return to_string(n);
  }
}

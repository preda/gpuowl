// Copyright (C) Mihai Preda.

#include "FFTConfig.h"
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

using namespace std;

struct FftBpw {
  string fft;
  array<double, 4> bpw;
};

map<string, array<double, 4>> BPW {
#include "fftbpw.h"
};

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

u32 FFTShape::getMaxCarry32(u32 exponent) const {
  u32 N = size();
  return (u32) (0x3242 * pow(2.0, 0.558 * log2(N / (1.25 * 1024 * 1024)) + double(exponent) / double(N) - 18.706));
}

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
// - a single config: 1K:13:256
// - a size: 6.5M
// - a range of sizes: 6.5M-7M
// - a list: 6M-7M,1K:13:256
vector<FFTShape> FFTShape::multiSpec(const string& iniSpec) {
  if (iniSpec.empty()) { return allShapes(); }

  vector<FFTShape> ret;

  for (const string &spec : split(iniSpec, ',')) {
    auto parts = split(spec, ':');
    assert(parts.size() <= 3);
    if (parts.size() == 3) {
      u32 width = parseInt(parts[0]);
      u32 middle = parseInt(parts[1]);
      u32 height = parseInt(parts[2]);
      ret.push_back({width, middle, height});
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
  for (u32 width : {256, 512, 1024, 4096}) {
    for (u32 height : {256, 512, 1024/*, 4096*/}) {
      for (u32 middle : {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}) {
        u32 sz = width * height * middle * 2;
        if (sizeFrom <= sz && sz <= sizeTo) {
          configs.push_back({width, middle, height});
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
  vector<string> v = split(spec, ':');
  assert(v.size() == 3);
  *this = FFTShape{parseInt(v.at(0)), parseInt(v.at(1)), parseInt(v.at(2))};
}

FFTShape::FFTShape(u32 w, u32 m, u32 h) :
  width{w}, middle{m}, height{h} {
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
      log("BPW info for %s not found, using defaults\n", s.c_str());
      double d = 0.275 * (log2(size()) - log2(256 * 13 * 1024 * 2));
      bpw = {18.1-d, 18.2-d, 18.2-d, 18.3-d};
    }
  }
}

FFTConfig::FFTConfig(const string& spec) :
  shape{spec.substr(0, spec.rfind(':'))},
  variant{parseInt(spec.substr(spec.rfind(':') + 1))}
{
  assert(variant < N_VARIANT);
}

FFTConfig FFTConfig::bestFit(u32 E, const string& spec) {
  // A FFT-spec was given, simply take the first FFT from the spec that can handle E
  if (!spec.empty()) {
    for (const FFTShape& shape : FFTShape::multiSpec(spec)) {
      for (u32 v = 0; v < 4; ++v) {
        if (FFTConfig fft{shape, v}; fft.maxExp() >= E) { return fft; }
      }
    }
    log("%s can not handle %u\n", spec.c_str(), E);
    throw "FFT size";
  }

  // No FFT-spec given, so choose from tune.txt the fastest FFT that can handle E
  vector<TuneEntry> tunes = TuneEntry::readTuneFile();
  for (const TuneEntry& e : tunes) {
    // The first acceptable is the best as they're sorted by cost
    if (E <= e.fft.maxExp()) { return e.fft; }
  }

  log("No FFTs found in tune.txt that can handle %u. Consider tuning with -tune\n", E);

  // Take the first FFT that can handle E
  for (const FFTShape& shape : FFTShape::allShapes()) {
    for (u32 v = 0; v < 4; ++v) {
      if (FFTConfig fft{shape, v}; fft.maxExp() >= E) { return fft; }
    }
  }

  log("No FFT found for %u\n", E);
  throw "No FFT";
}


string numberK(u32 n) {
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

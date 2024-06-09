// Copyright (C) Mihai Preda.

#include "FFTConfig.h"
#include "common.h"
#include "log.h"
#include "File.h"
#include "Args.h"

#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>

using namespace std;

vector<FFT> FFTShape::readTune() {
  vector<FFT> ret;

  File fi = File::openRead("tune.txt");
  if (!fi) { return {}; }
  for (string line : fi) {
    if (line.empty() || line[0] == '#') { continue; }
    u32 priority{};
    u32 maxExp{};
    float bpw{};
    int pos = 0;
    int nScan = sscanf(line.c_str(), "%u %u %f : %n", &priority, &maxExp, &bpw, &pos);
    if (!pos || nScan < 3) {
      log("Invalid tune line \"%s\" ignored\n", line.c_str());
      continue;
    }
    string tail = line.substr(pos);
    string fftSpec;
    vector<KeyVal> uses;
    for (const auto& [k, v] : Args::splitArgLine(tail)) {
      if (k == "-fft") {
        fftSpec = v;
      } else if (k == "-uses") {
        uses = Args::splitUses(v);
      } else {
        log("Unexpeted %s %s\n", k.c_str(), v.c_str());
      }
    }
    assert(!fftSpec.empty());
    ret.push_back({priority, maxExp, fftSpec, uses});
  }
  std::sort(ret.begin(), ret.end(), [](const FFT& a, const FFT& b) { return a.priority < b.priority; });
  return ret;
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

u32 FFTShape::getMaxCarry32(u32 fftSize, u32 exponent) {
  return (u32) (0x3242 * pow (2.0, 0.558 * log2(fftSize / (1.25 * 1024 * 1024)) + double(exponent) / double(fftSize) - 18.706));
}

namespace {

u32 parseInt(const string& s) {
  if (s.empty()) { return 1; }
  char c = s.back();
  u32 multiple = c == 'k' || c == 'K' ? 1024 : c == 'm' || c == 'M' ? 1024 * 1024 : 1;
  return strtod(s.c_str(), nullptr) * multiple;
}

vector<FFTShape> specsForSize(u32 fftSize) {
  vector<FFTShape> ret;
  for (FFTShape& c : FFTShape::genConfigs()) {
    if (c.fftSize() == fftSize) { ret.push_back(c); }
  }
  return ret;
}

} // namespace

FFTShape FFTShape::fromSpec(const string& spec) {
  return multiSpec(spec).front();
}

// Accepts:
// - a single config e.g. "1K:13:256"
// - a size e.g. "6.5M"
// - a range e.g. "6M-7M"
vector<FFTShape> FFTShape::multiSpec(const string& spec) {
  assert(!spec.empty());
  auto pDash = spec.find('-');
  if (pDash != string::npos) {
    string from = spec.substr(0, pDash);
    string to = spec.substr(pDash + 1);
    u32 sizeFrom = multiSpec(from).front().fftSize();
    u32 sizeTo = multiSpec(to).front().fftSize();
    auto all = genConfigs();
    vector<FFTShape> ret;
    for (const auto& c : all) {
      if (c.fftSize() >= sizeFrom && c.fftSize() <= sizeTo) { ret.push_back(c); }
    }
    return ret;
  }

  bool hasParts = spec.find(':') != string::npos;
  if (hasParts) {
    auto p1 = spec.find(':');
    u32 width = parseInt(spec.substr(0, p1));
    auto p2 = spec.find(':', p1+1);
    if (p2 == string::npos) {
      log("FFT spec must be of the form width:middle:height , found '%s'\n", spec.c_str());
      throw "Invalid FFT spec";
    }
    u32 middle = parseInt(spec.substr(p1+1, p2 - (p1 + 1)));
    u32 height = parseInt(spec.substr(p2+1));
    return {{width, middle, height}};
  } else {
    u32 fftSize = parseInt(spec);
    auto specs = specsForSize(fftSize);
    if (specs.empty()) {
      log("Could not find a FFT config for '%s'\n", spec.c_str());
      throw "Invalid FFT spec";
    }
    return specs;
  }
}

vector<FFTShape> FFTShape::genConfigs() {
  vector<FFTShape> configs;
  for (u32 width : {256, 512, 1024, 4096}) {
    for (u32 height : {256, 512, 1024}) {
      for (u32 middle : {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}) {
        configs.push_back({width, middle, height});
      }
    }
  }
  std::sort(configs.begin(), configs.end(),
            [](const FFTShape &a, const FFTShape &b) {
              if (a.fftSize() != b.fftSize()) { return (a.fftSize() < b.fftSize()); }
              if (a.width != b.width) {
                if (a.width == 1024 || b.width == 1024) { return a.width == 1024; }
                return a.width < b.width;
              }
              return a.height < b.height;
            });
  return configs;
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

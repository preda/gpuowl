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

u32 FFTShape::getMaxCarry32(u32 exponent) const {
  u32 N = fftSize();
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
  if (spec.empty()) { return genConfigs(); }

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

FFTShape::FFTShape(u32 w, u32 m, u32 h) :
  width{w}, middle{m}, height{h} {
  string s = spec();
  bpw = BPW[s];
}

FFTConfig::FFTConfig(const string& spec) :
  shape{FFTShape::fromSpec(spec.substr(0, spec.rfind(':')))},
  variant{parseInt(spec.substr(spec.rfind(':') + 1))}
{
  assert(variant < N_VARIANT);
}

vector<TuneEntry> readTuneFile() {
  File tuneFile = File::openRead("tune.txt");
  if (!tuneFile) { return {}; }
  vector<TuneEntry> v;

  double cost;
  char fftSpecBuf[64];
  char configBuf[128];
  for (string line : tuneFile) {
    *fftSpecBuf = 0;
    *configBuf = 0;
    int n = sscanf(line.c_str(), "%lf %63s %127s", &cost, fftSpecBuf, configBuf);
    if (n < 3) {
      log("Could not parse tune.txt line '%s'\n", line.c_str());
      break;
    }
    v.push_back({cost, FFTConfig{fftSpecBuf}, configBuf});
  }
  std::sort(v.begin(), v.end(), [](const TuneEntry& a, const TuneEntry& b) { return a.cost < b.cost; });
  return v;
}

bool FFTConfig::matches(const string& spec) const {
  if (spec.empty()) { return true; }

  // An interval of sizes
  auto pDash = spec.find('-');
  if (pDash != string::npos) {
    u32 mySize = shape.fftSize();

    string from = spec.substr(0, pDash);
    string to = spec.substr(pDash + 1);
    u32 sizeFrom = FFTShape::multiSpec(from).front().fftSize();
    if (mySize < sizeFrom) { return false; }
    u32 sizeTo = FFTShape::multiSpec(to).front().fftSize();
    if (mySize > sizeTo) { return false; }
    return true;
  }

  bool hasParts = spec.find(':') != string::npos;
  if (hasParts) {
    auto p1 = spec.find(':');
    string widthStr = spec.substr(0, p1);
    if (!widthStr.empty() && parseInt(widthStr) != shape.width) { return false; }
    auto p2 = spec.find(':', p1+1);
    if (p2 == string::npos) {
      log("FFT spec must be of the form width:middle:height , found '%s'\n", spec.c_str());
      throw "Invalid FFT spec";
    }

    string middleStr = spec.substr(p1+1, p2 - (p1 + 1));
    if (!middleStr.empty() && parseInt(middleStr) != shape.middle) { return false; }

    string heightStr = spec.substr(p2+1);
    if (!heightStr.empty() && parseInt(heightStr) != shape.height) { return false; }

    auto p3 = spec.find(':', p2 + 1);
    if (p3 != string::npos && variant != parseInt(spec.substr(p3+1))) { return false; }
    return true;
  } else {
    return shape.fftSize() == parseInt(spec);
  }
}

pair<FFTConfig, string> FFTConfig::bestFit(u32 E, const string& fftSpec) {
  // Choose from tune.txt the fastest FFT that's acceptable according to maxExp and fftSpec.
  vector<TuneEntry> tunes = readTuneFile();
  for (const TuneEntry& e : tunes) {
    // The first acceptable is the best as they're sorted by cost
    if (E <= e.fft.maxExp() && e.fft.matches(fftSpec)) { return {e.fft, e.config}; }
  }

  log("No acceptable entries were found in tune.txt. Consider tuning (-tune)\n");

  vector<FFTShape> candidates = FFTShape::multiSpec(fftSpec);
  for (const FFTShape& shape : candidates) {
    for (u32 v = 0; v < 4; ++v) {
      FFTConfig fft{shape, v};
      if (fft.maxExp() >= E) { return {fft, {}}; }
    }
  }

  log("No FFT found for %u given '%s'\n", E, fftSpec.c_str());
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

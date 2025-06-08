// Copyright (C) Mihai Preda

#include "tune.h"
#include "Args.h"
#include "FFTConfig.h"
#include "Gpu.h"
#include "GpuCommon.h"
#include "Primes.h"
#include "log.h"
#include "File.h"
#include "TuneEntry.h"

#include <numeric>
#include <string>
#include <vector>
#include <cassert>

using std::accumulate;

using namespace std;

vector<string> split(const string& s, char delim) {
  vector<string> ret;
  size_t start = 0;
  while (true) {
    size_t p = s.find(delim, start);
    if (p == string::npos) {
      ret.push_back(s.substr(start));
      break;
    } else {
      ret.push_back(s.substr(start, p - start));
    }
    start = p + 1;
  }
  return ret;
}

namespace {

vector<TuneConfig> permute(const vector<pair<string, vector<string>>>& params) {
  vector<TuneConfig> configs;

  int n = params.size();
  vector<int> vpos(n);
  while (true) {
    TuneConfig config;
    for (int i = 0; i < n; ++i) {
      config.push_back({params[i].first, params[i].second[vpos[i]]});
    }
    configs.push_back(config);

    int i;
    for (i = n-1; i >= 0; --i) {
      if (vpos[i] < int(params[i].second.size()) - 1) {
        ++vpos[i];
        break;
      } else {
        vpos[i] = 0;
      }
    }

    if (i < 0) { return configs; }
  }
}

vector<TuneConfig> getTuneConfigs(const string& tune) {
  vector<pair<string, vector<string>>> params;
  for (auto& part : split(tune, ';')) {
    auto keyVal = split(part, '=');
    assert(keyVal.size() == 2);
    string key = keyVal.front();
    string val = keyVal.back();
    params.push_back({key, split(val, ',')});
  }
  return permute(params);
}

string toString(TuneConfig config) {
  string s{};
  for (const auto& [k, v] : config) { s += k + '=' + v + ','; }
  s.pop_back();
  return s;
}

struct Entry {
  FFTShape shape;
  TuneConfig config;
  double cost;
};

string formatEntry(Entry e) {
  char buf[256];
  snprintf(buf, sizeof(buf), "! %s %s # %.0f\n",
           e.shape.spec().c_str(), toString(e.config).c_str(), e.cost);
  return buf;
}

string formatConfigResults(const vector<Entry>& results) {
  string s;
  for (const Entry& e : results) { if (e.shape.width) { s += formatEntry(e); } }
  return s;
}

} // namespace

double Tune::maxBpw(FFTConfig fft) {

//  double bpw = oldBpw;

  const double TARGET = 28;
  const u32 sample_size = 5;

  // Estimate how much bpw needs to change to increase/decrease Z by 1.
  // This doesn't need to be a very accurate estimate.
  // This estimate comes from analyzing a 4M FFT and a 7.5M FFT.
  // The 4M FFT needed a .015 step, the 7.5M FFT needed a .012 step.
  double bpw_step = .015 + (log2(fft.size()) - log2(4.0*1024*1024)) / (log2(7.5*1024*1024) - log2(4.0*1024*1024)) * (.012 - .015);

  // Pick a bpw that might be close to Z=34, it is best to err on the high side of Z=34
  double bpw1 = fft.maxBpw() - 9 * bpw_step;                                      // Old bpw gave Z=28, we want Z=34 (or more)

// The code below was used when building the maxBpw table from scratch
//  u32   non_best_width = N_VARIANT_W - 1 - variant_W(fft.variant);              // Number of notches below best-Z width variant
//  u32   non_best_middle = N_VARIANT_M - 1 - variant_M(fft.variant);             // Number of notches below best-Z middle variant
//  double bpw1 = 18.3 - 0.275 * (log2(fft.size()) - log2(256 * 13 * 1024 * 2)) - // Default max bpw from an old gpuowl version
//              9 * bpw_step -                                                    // Default above should give Z=28, we want Z=34 (or more)
//                (.08/.012 * bpw_step) * non_best_width -                        // 7.5M FFT has ~.08 bpw difference for each width variant below best variant
//                (.06 + .04 * (fft.shape.middle - 4) / 11) * non_best_middle;    // Assume .1 bpw difference MIDDLE=15 and .06 for MIDDLE=4
//Above fails for FFTs below 512K.  Perhaps we should ditch the above and read from the existing fftbpw.h data to get our starting guess.
//if (fft.size() < 512000) bpw1 = 19, bpw_step = .02;

  // Fine tune our estimate for Z=34
  double z1 = zForBpw(bpw1, fft, 1);
printf ("Guess bpw for %s is %.2f first Z34 is %.2f\n", fft.spec().c_str(), bpw1, z1);
  while (z1 < 31.0 || z1 > 37.0) {
    double prev_bpw1 = bpw1;
    double prev_z1 = z1;
    bpw1 = bpw1 + (z1 - 34) * bpw_step;
    z1 = zForBpw(bpw1, fft, 1);
printf ("Reguess bpw for %s is %.2f first Z34 is %.2f\n", fft.spec().c_str(), bpw1, z1);
    bpw_step = - (bpw1 - prev_bpw1) / (z1 - prev_z1);
    if (bpw_step < 0.005) bpw_step = 0.005;
    if (bpw_step > 0.025) bpw_step = 0.025;
  }

  // Get more samples for this bpw -- average in the sample we already have
  z1 = (z1 + (sample_size - 1) * zForBpw(bpw1, fft, sample_size - 1)) / sample_size;

  // Pick a bpw somewhere near Z=22 then fine tune the guess
  double bpw2 = bpw1 + (z1 - 22) * bpw_step;
  double z2 = zForBpw(bpw2, fft, 1);
printf ("Guess bpw for %s is %.2f first Z22 is %.2f\n", fft.spec().c_str(), bpw2, z2);
  while (z2 < 20.0 || z2 > 25.0) {
    double prev_bpw2 = bpw2;
    double prev_z2 = z2;
//    bool error_recovery = (z2 <= 0.0);
//    if (error_recovery) bpw2 -= bpw_step; else
    bpw2 = bpw2 + (z2 - 21) * bpw_step;
    z2 = zForBpw(bpw2, fft, 1);
printf ("Reguess bpw for %s is %.2f first Z22 is %.2f\n", fft.spec().c_str(), bpw2, z2);
//  if (error_recovery) { if (z2 >= 20.0) break; else continue; }
    bpw_step = - (bpw2 - prev_bpw2) / (z2 - prev_z2);
    if (bpw_step < 0.005) bpw_step = 0.005;
    if (bpw_step > 0.025) bpw_step = 0.025;
  }

  // Get more samples for this bpw -- average in the sample we already have
  z2 = (z2 + (sample_size - 1) * zForBpw(bpw2, fft, sample_size - 1)) / sample_size;

  // Interpolate for the TARGET Z value
  return bpw2 + (bpw1 - bpw2) * (TARGET - z2) / (z1 - z2);
}

double Tune::zForBpw(double bpw, FFTConfig fft, u32 count) {
  u32 exponent = (count == 1) ? primes.prevPrime(fft.size() * bpw) : primes.nextPrime(fft.size() * bpw);
  double total_z = 0.0;
  for (u32 i = 0; i < count; i++, exponent = primes.nextPrime (exponent + 1)) {
    auto [ok, res, roeSq, roeMul] = Gpu::make(q, exponent, shared, fft, {}, false)->measureROE(true);
    double z = roeSq.z();
    total_z += z;
log("Zforbpw %.2f (z %.2f) : %s\n", bpw, z, fft.spec().c_str());
    if (!ok) { log("Error at bpw %.2f (z %.2f) : %s\n", bpw, z, fft.spec().c_str()); continue; }
  }
//printf ("Out zForBpw %s %.2f avg %.2f\n", fft.spec().c_str(), bpw, total_z / count);
  return total_z / count;
}

void Tune::ztune() {
  File ztune = File::openAppend("ztune.txt");
  ztune.printf("\n// %s\n\n", shortTimeStr().c_str());

  // Study a specific shape and variant
  if (0) {
    FFTShape shape = FFTShape(512, 15, 512);
    u32 variant = 202;
    u32 sample_size = 5;
    FFTConfig fft{shape, variant, CARRY_AUTO};
    for (double bpw = 18.18; bpw < 18.305; bpw += 0.02) {
      double z = zForBpw(bpw, fft, sample_size);
      log ("Avg zForBpw %s %.2f %.2f\n", fft.spec().c_str(), bpw, z);
    }
  }

  // Generate a decent-sized sample that correlates bpw and Z in a range that is close to the target Z value of 28.
  // For no particularly good reason, I strive to find the bpw for Z values near 35 and 21.
  // Over this narrow Z range, linear curve fit should work well.  The Z data is noisy, so more samples is better.

  auto configs = FFTShape::multiSpec(shared.args->fftSpec);
  for (FFTShape shape : configs) {

    // 4K widths store data on variants 100, 101, 202, 110, 111, 212
    u32 bpw_variants[NUM_BPW_ENTRIES] = {000, 101, 202, 10, 111, 212};
    if (shape.width > 1024) bpw_variants[0] = 100, bpw_variants[3] = 110;

    // Copy the existing bpw array (in case we're replacing only some of the entries)
    array<double, NUM_BPW_ENTRIES> bpw;
    bpw = shape.bpw;

    // Not all shapes have their maximum bpw per-computed.  But one can work on a non-favored shape by specifying it on the command line.
    if (configs.size() > 1) {
      if (!shape.isFavoredShape()) { log ("Skipping %s\n", shape.spec().c_str()); continue; }
    }

    // Test specific variants needed for the maximum bpw table in fftbpw.h
    for (u32 j = 0; j < NUM_BPW_ENTRIES; ++j) {
      FFTConfig fft{shape, bpw_variants[j], CARRY_AUTO};
      bpw[j] = maxBpw(fft);
    }
    string s = "\""s + shape.spec() + "\"";
//    ztune.printf("{%12s, {%.3f, %.3f, %.3f, %.3f, %.3f, %.3f}},\n", s.c_str(), bpw[0], bpw[1], bpw[2], bpw[3], bpw[4], bpw[5]);
    ztune.printf("{%12s, {", s.c_str());
    for (u32 j = 0; j < NUM_BPW_ENTRIES; ++j) ztune.printf("%s%.3f", j ? ", " : "", bpw[j]);
    ztune.printf("}},\n");
  }
}

void Tune::carryTune() {
  File fo = File::openAppend("carrytune.txt");
  fo.printf("\n// %s\n\n", shortTimeStr().c_str());
  shared.args->flags["STATS"] = "1";
  u32 prevSize = 0;
  for (FFTShape shape : FFTShape::multiSpec(shared.args->fftSpec)) {
    FFTConfig fft{shape, LAST_VARIANT, CARRY_AUTO};
    if (prevSize == fft.size()) { continue; }
    prevSize = fft.size();

    vector<double> zv;
    double m = 0;
    const double mid = fft.shape.carry32BPW();
    for (double bpw : {mid - 0.05, mid + 0.05}) {
      u32 exponent = primes.nearestPrime(fft.size() * bpw);
      auto [ok, carry] = Gpu::make(q, exponent, shared, fft, {}, false)->measureCarry();
      m = carry.max;
      if (!ok) { log("Error %s at %f\n", fft.spec().c_str(), bpw); }
      zv.push_back(carry.z());
    }

    double avg = (zv[0] + zv[1]) / 2;
    u32 exponent = fft.shape.carry32BPW() * fft.size();
    double pErr100 = -expm1(-exp(-avg) * exponent * 100);
    log("%14s %.3f : %.3f (%.3f %.3f) %f %.0f%%\n", fft.spec().c_str(), mid, avg, zv[0], zv[1], m, pErr100 * 100);
    fo.printf("%f %f\n", log2(fft.size()), avg);
  }
}

template<typename T>
void add(vector<T>& a, const vector<T>& b) {
  a.insert(a.end(), b.begin(), b.end());
}

void Tune::ctune() {
  Args *args = shared.args;

  vector<string> ctune = args->ctune;
  if (ctune.empty()) { ctune.push_back("IN_WG=256,128,64;IN_SIZEX=32,16,8;OUT_WG=256,128,64;OUT_SIZEX=32,16,8"); }

  vector<vector<TuneConfig>> configsVect;
  for (const string& s : ctune) {
    configsVect.push_back(getTuneConfigs(s));
  }

  vector<Entry> results;

  auto shapes = FFTShape::multiSpec(args->fftSpec);
  {
    string str;
    for (const auto& s : shapes) { str += s.spec() + ','; }
    if (!str.empty()) { str.pop_back(); }
    log("FFTs: %s\n", str.c_str());
  }

  for (FFTShape shape : shapes) {
    FFTConfig fft{shape, 0, CARRY_32};
    u32 exponent = primes.prevPrime(fft.maxExp());
    // log("tuning %10s with exponent %u\n", fft.shape.spec().c_str(), exponent);

    vector<int> bestPos(configsVect.size());
    Entry best{{1, 1, 1}, {}, 1e9};

    for (u32 i = 0; i < configsVect.size(); ++i) {
      for (u32 pos = i ? 1 : 0; pos < configsVect[i].size(); ++pos) {
        vector<KeyVal> c;

        for (u32 k = 0; k < i; ++k) {
          add(c, configsVect[k][bestPos[k]]);
        }
        add(c, configsVect[i][pos]);
        for (u32 k = i + 1; k < configsVect.size(); ++k) {
          add(c, configsVect[k][bestPos[k]]);
        }
        auto cost = Gpu::make(q, exponent, shared, fft, c, false)->timePRP();

        bool isBest = (cost < best.cost);
        if (isBest) {
          bestPos[i] = pos;
          best = {shape, c, cost};
        }
        log("%c %6.0f : %s %s\n",
            isBest ? '*' : ' ', cost, shape.spec().c_str(), toString(c).c_str());
      }
    }
    results.push_back(best);
    log("%s", formatEntry(best).c_str());
  }
  log("\nBest configs (lines can be copied to config.txt):\n%s", formatConfigResults(results).c_str());
}

void Tune::tune() {
  Args *args = shared.args;
  string fftSpec = args->fftSpec;

//GW: detail all the configs we should auto-time first

  // Flags that prune the amount of shapes and variants to time.
  // These should be computed automatically and saved in the tune.txt or config.txt file.
  // Tune.txt file should have a version number.

  // A command line option to run more combinations (higher number skips more combos)
  int skip_some_WH_variants = 1;                // 0 = skip nothing, 1 = skip slower widths/heights unless they have better Z, 2 = only run fastest widths/heights

  // The width = height = 512 FFT shape is so good, we probably don't need to time the width = 1024, height = 256 shape.
  bool skip_1K_256 = 1;

  // There are some variands only AMD GPUs can execute
  bool AMDGPU = isAmdGpu(q->context->deviceId());

// make command line args for this?
skip_some_WH_variants = 2;
skip_1K_256 = 0;

//GW:  Suggest tuning with TAIL_KERNELS=2 even if production runs use TAIL_KERNELS=3

  // For each width, time the 001, 101, and 201 variants to find the fastest width variant.
  // In an ideal world we'd use the -time feature and look at the kCarryFused timing.  Then we'd save this info in config.txt or tune.txt.
  map<int, u32> fastest_width_variants;

  // For each height, time the 100, 101, and 102 variants to find the fastest height variant.
  // In an ideal world we'd use the -time feature and look at the tailSquare timing.  Then we'd save this info in config.txt or tune.txt.
  map<int, u32> fastest_height_variants;

  vector<TuneEntry> results = TuneEntry::readTuneFile(*args);
  vector<FFTShape> shapes = FFTShape::multiSpec(args->fftSpec);

  // Loop through all possible FFT shapes
  for (const FFTShape& shape : shapes) {

    // Time an exponent that's good for all variants and carry-config.
    u32 exponent = primes.prevPrime(FFTConfig{shape, shape.width <= 1024 ? 0u : 100u, CARRY_32}.maxExp());

  // Loop through all possible variants
    for (u32 variant = 0; variant <= LAST_VARIANT; variant = next_variant (variant)) {

      // Only AMD GPUs support variant zero (BCAST) and only if width <= 1024.
      if (variant_W(variant) == 0) {
        if (!AMDGPU) continue;
        if (shape.width > 1024) continue;
      }

      // Only AMD GPUs support variant zero (BCAST) and only if height <= 1024.
      if (variant_H(variant) == 0) {
        if (!AMDGPU) continue;
        if (shape.height > 1024) continue;
      }

      // If only one shape was specified on the command line, time it.  This lets the user time any shape, including non-favored ones.
      if (shapes.size() > 1) {

        // Skip less-favored shapes
        if (!shape.isFavoredShape()) continue;

        // Skip width = 1K, height = 256
        if (shape.width == 1024 && shape.height == 256 && skip_1K_256) continue;

        // Skip variants where width or height are not using the fastest variant.
        // NOTE: We ought to offer a tune=option where we also test more accurate variants to extend the FFT's max exponent.
        if (skip_some_WH_variants) {
          u32 fastest_width = 1;
          if (auto it = fastest_width_variants.find(shape.width); it != fastest_width_variants.end()) {
            fastest_width = it->second;
          } else {
            FFTShape test = FFTShape(shape.width, 12, 256);
            double cost, min_cost = -1.0;
            for (u32 w = 0; w < N_VARIANT_W; w++) {
              if (w == 0 && !AMDGPU) continue;
              if (w == 0 && test.width > 1024) continue;
              FFTConfig fft{test, variant_WMH (w, 0, 1), CARRY_32};
              cost = Gpu::make(q, primes.prevPrime(fft.maxExp()), shared, fft, {}, false)->timePRP();
              log("Fast width search %6.1f %12s\n", cost, fft.spec().c_str());
              if (min_cost < 0.0 || cost < min_cost) { min_cost = cost; fastest_width = w; }
            }
            fastest_width_variants[shape.width] = fastest_width;
          }
          if (skip_some_WH_variants == 2 && variant_W(variant) != fastest_width) continue;
          if (skip_some_WH_variants == 1 &&
              FFTConfig{shape, variant, CARRY_32}.maxBpw() <
                  FFTConfig{shape, variant_WMH (fastest_width, variant_M(variant), variant_H(variant)), CARRY_32}.maxBpw()) continue;
        }
        if (skip_some_WH_variants) {
          u32 fastest_height = 1;
          if (auto it = fastest_height_variants.find(shape.height); it != fastest_height_variants.end()) {
            fastest_height = it->second;
          } else {
            FFTShape test = FFTShape(shape.height, 12, shape.height);
            double cost, min_cost = -1.0;
            for (u32 h = 0; h < N_VARIANT_H; h++) {
              if (h == 0 && !AMDGPU) continue;
              if (h == 0 && test.height > 1024) continue;
              FFTConfig fft{test, variant_WMH (1, 0, h), CARRY_32};
              cost = Gpu::make(q, primes.prevPrime(fft.maxExp()), shared, fft, {}, false)->timePRP();
              log("Fast height search %6.1f %12s\n", cost, fft.spec().c_str());
              if (min_cost < 0.0 || cost < min_cost) { min_cost = cost; fastest_height = h; }
            }
            fastest_height_variants[shape.height] = fastest_height;
          }
          if (skip_some_WH_variants == 2 && variant_H(variant) != fastest_height) continue;
          if (skip_some_WH_variants == 1 &&
              FFTConfig{shape, variant, CARRY_32}.maxBpw() <
                  FFTConfig{shape, variant_WMH (variant_W(variant), variant_M(variant), fastest_height), CARRY_32}.maxBpw()) continue;
        }
      }

//GW: If variant is specified on command line, time it (and only it)??  Or an option to only time one variant number??

      vector carryToTest{CARRY_32};
      // We need to test both carry-32 and carry-64 only when the carry transition is within the BPW range.
      if (FFTConfig{shape, variant, CARRY_64}.maxBpw() > FFTConfig{shape, variant, CARRY_32}.maxBpw()) {
        carryToTest.push_back(CARRY_64);
      }

      for (auto carry : carryToTest) {
        FFTConfig fft{shape, variant, carry};

        // Skip middle = 1, CARRY_32 if maximum exponent would be the same as middle = 0, CARRY_32
        if (variant_M(variant) > 0 && carry == CARRY_32 && fft.maxExp() <= FFTConfig{shape, variant - 10, CARRY_32}.maxExp()) continue;

        double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
        bool isUseful = TuneEntry{cost, fft}.update(results);
        log("%c %6.1f %12s %9u\n", isUseful ? '*' : ' ', cost, fft.spec().c_str(), fft.maxExp());
      }
    }
  }

  TuneEntry::writeTuneFile(results);
}

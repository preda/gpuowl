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
  vector<FFTShape> shapes = FFTShape::multiSpec(args->fftSpec);

  // There are some options and variants that are different based on GPU manufacturer
  bool AMDGPU = isAmdGpu(q->context->deviceId());

  // Look for best settings of various options

  if (1) {
    u32 variant = 101;
//GW: if fft spec on the command line specifies a variant then we should use that variant (I get some interesting results with 000 vs 101 vs 201 vs 202 likely due to rocm optimizer)

    // Find best IN_WG,IN_SIZEX setting
    if (1) {
      const FFTShape& shape = shapes[0];
      FFTConfig fft{shape, variant, CARRY_32};
      u32 exponent = primes.prevPrime(fft.maxExp());
      u32 best_in_wg = 0;
      u32 best_in_sizex = 0;
      double best_cost = -1.0;
      for (u32 in_wg : {64, 128, 256}) {
        for (u32 in_sizex : {8, 16, 32}) {
          shared.args->flags["IN_WG"] = to_string(in_wg);
          shared.args->flags["IN_SIZEX"] = to_string(in_sizex);
          double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
          log("Time for %12s using IN_WG=%u, IN_SIZEX=%u is %6.1f\n", fft.spec().c_str(), in_wg, in_sizex, cost);
	  if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_in_wg = in_wg; best_in_sizex = in_sizex; }
	}
      }
      log("Best IN_WG, IN_SIZEX is %u, %u.  Default is 128, 16.\n", best_in_wg, best_in_sizex);
      shared.args->flags["IN_WG"] = to_string(best_in_wg);
      shared.args->flags["IN_SIZEX"] = to_string(best_in_sizex);
    }

    // Find best OUT_WG,OUT_SIZEX setting
    if (1) {
      const FFTShape& shape = shapes[0];
      FFTConfig fft{shape, variant, CARRY_32};
      u32 exponent = primes.prevPrime(fft.maxExp());
      u32 best_out_wg = 0;
      u32 best_out_sizex = 0;
      double best_cost = -1.0;
      for (u32 out_wg : {64, 128, 256}) {
        for (u32 out_sizex : {8, 16, 32}) {
          shared.args->flags["OUT_WG"] = to_string(out_wg);
          shared.args->flags["OUT_SIZEX"] = to_string(out_sizex);
          double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
          log("Time for %12s using OUT_WG=%u, OUT_SIZEX=%u is %6.1f\n", fft.spec().c_str(), out_wg, out_sizex, cost);
	  if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_out_wg = out_wg; best_out_sizex = out_sizex; }
	}
      }
      log("Best OUT_WG, OUT_SIZEX is %u, %u.  Default is 128, 16.\n", best_out_wg, best_out_sizex);
      shared.args->flags["OUT_WG"] = to_string(best_out_wg);
      shared.args->flags["OUT_SIZEX"] = to_string(best_out_sizex);
    }

    // Find best FAST_BARRIER setting
    if (1 && AMDGPU) {
      const FFTShape& shape = shapes[0];
      FFTConfig fft{shape, variant, CARRY_32};
      u32 exponent = primes.prevPrime(fft.maxExp());
      u32 best_fast_barrier = 0;
      double best_cost = -1.0;
      for (u32 fast_barrier : {0, 1}) {
        shared.args->flags["FAST_BARRIER"] = to_string(fast_barrier);
        double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
        log("Time for %12s using FAST_BARRIER=%u is %6.1f\n", fft.spec().c_str(), fast_barrier, cost);
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_fast_barrier = fast_barrier; }
      }
      log("Best FAST_BARRIER is %u.  Default FAST_BARRIER is 0.\n", best_fast_barrier);
      shared.args->flags["FAST_BARRIER"] = to_string(best_fast_barrier);
    }

    // Find best TAIL_TRIGS setting
    if (1) {
      const FFTShape& shape = shapes[0];
      FFTConfig fft{shape, variant, CARRY_32};
      u32 exponent = primes.prevPrime(fft.maxExp());
      u32 best_tail_trigs = 0;
      double best_cost = -1.0;
      for (u32 tail_trigs : {0, 1, 2}) {
        shared.args->flags["TAIL_TRIGS"] = to_string(tail_trigs);
        double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
        log("Time for %12s using TAIL_TRIGS=%u is %6.1f\n", fft.spec().c_str(), tail_trigs, cost);
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_tail_trigs = tail_trigs; }
      }
      log("Best TAIL_TRIGS is %u.  Default TAIL_TRIGS is 2.\n", best_tail_trigs);
      shared.args->flags["TAIL_TRIGS"] = to_string(best_tail_trigs);
    }

    // Find best TAIL_KERNELS setting
    if (1) {
      const FFTShape& shape = shapes[0];
      FFTConfig fft{shape, variant, CARRY_32};
      u32 exponent = primes.prevPrime(fft.maxExp());
      u32 best_tail_kernels = 0;
      double best_cost = -1.0;
      for (u32 tail_kernels : {0, 1, 2, 3}) {
        shared.args->flags["TAIL_KERNELS"] = to_string(tail_kernels);
        double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
        log("Time for %12s using TAIL_KERNELS=%u is %6.1f\n", fft.spec().c_str(), tail_kernels, cost);
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_tail_kernels = tail_kernels; }
      }
      if (best_tail_kernels & 1)
        log("Best TAIL_KERNELS is %u.  Default TAIL_KERNELS is 2.\n", best_tail_kernels);
      else
        log("Best TAIL_KERNELS is %u (but best may be %u when running two workers on one GPU).  Default TAIL_KERNELS is 2.\n", best_tail_kernels, best_tail_kernels | 1);
      shared.args->flags["TAIL_KERNELS"] = to_string(best_tail_kernels);
    }

    // Find best TABMUL_CHAIN setting
    if (1) {
      const FFTShape& shape = shapes[0];
      FFTConfig fft{shape, 101, CARRY_32};
      u32 exponent = primes.prevPrime(fft.maxExp());
      u32 best_tabmul_chain = 0;
      double best_cost = -1.0;
      for (u32 tabmul_chain : {0, 1}) {
        shared.args->flags["TABMUL_CHAIN"] = to_string(tabmul_chain);
        double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
        log("Time for %12s using TABMUL_CHAIN=%u is %6.1f\n", fft.spec().c_str(), tabmul_chain, cost);
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_tabmul_chain = tabmul_chain; }
      }
      log("Best TABMUL_CHAIN is %u.  Default TABMUL_CHAIN is 0.\n", best_tabmul_chain);
      shared.args->flags["TABMUL_CHAIN"] = to_string(best_tabmul_chain);
    }

    // Find best PAD setting.  Default is 256 bytes for AMD, 0 for all others.
    if (1) {
      const FFTShape& shape = shapes[0];
      FFTConfig fft{shape, variant, CARRY_32};
      u32 exponent = primes.prevPrime(fft.maxExp());
      u32 best_pad = 0;
      double best_cost = -1.0;
      for (u32 pad : {0, 64, 128, 256, 512}) {
        shared.args->flags["PAD"] = to_string(pad);
        double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
        log("Time for %12s using PAD=%u is %6.1f\n", fft.spec().c_str(), pad, cost);
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_pad = pad; }
      }
      log("Best PAD is %u bytes.  Default PAD is %u bytes.\n", best_pad, AMDGPU ? 256 : 0);
      shared.args->flags["PAD"] = to_string(best_pad);
    }

    // Find best NONTEMPORAL setting
    if (1) {
      const FFTShape& shape = shapes[0];
      FFTConfig fft{shape, variant, CARRY_32};
      u32 exponent = primes.prevPrime(fft.maxExp());
      u32 best_nontemporal = 0;
      double best_cost = -1.0;
      for (u32 nontemporal : {0, 1}) {
        shared.args->flags["NONTEMPORAL"] = to_string(nontemporal);
        double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
        log("Time for %12s using NONTEMPORAL=%u is %6.1f\n", fft.spec().c_str(), nontemporal, cost);
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_nontemporal = nontemporal; }
      }
      log("Best NONTEMPORAL is %u.  Default NONTEMPORAL is 0.\n", best_nontemporal);
      shared.args->flags["NONTEMPORAL"] = to_string(best_nontemporal);
    }

    // Find best UNROLL_W setting
    if (1) {
      const FFTShape& shape = shapes[0];
      FFTConfig fft{shape, variant, CARRY_32};
      u32 exponent = primes.prevPrime(fft.maxExp());
      u32 best_unroll_w = 0;
      double best_cost = -1.0;
      for (u32 unroll_w : {0, 1}) {
        shared.args->flags["UNROLL_W"] = to_string(unroll_w);
        double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
        log("Time for %12s using UNROLL_W=%u is %6.1f\n", fft.spec().c_str(), unroll_w, cost);
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_unroll_w = unroll_w; }
      }
      log("Best UNROLL_W is %u.  Default UNROLL_W is %u.\n", best_unroll_w, AMDGPU ? 0 : 1);
      shared.args->flags["UNROLL_W"] = to_string(best_unroll_w);
    }

    // Find best UNROLL_H setting
    if (1) {
      const FFTShape& shape = shapes[0];
      FFTConfig fft{shape, variant, CARRY_32};
      u32 exponent = primes.prevPrime(fft.maxExp());
      u32 best_unroll_h = 0;
      double best_cost = -1.0;
      for (u32 unroll_h : {0, 1}) {
        shared.args->flags["UNROLL_H"] = to_string(unroll_h);
        double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
        log("Time for %12s using UNROLL_H=%u is %6.1f\n", fft.spec().c_str(), unroll_h, cost);
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_unroll_h = unroll_h; }
      }
      log("Best UNROLL_H is %u.  Default UNROLL_H is %u.\n", best_unroll_h, AMDGPU && shape.height >= 1024 ? 0 : 1);
      shared.args->flags["UNROLL_H"] = to_string(best_unroll_h);
    }

    // Find best ZEROHACK_W setting
    if (1) {
      const FFTShape& shape = shapes[0];
      FFTConfig fft{shape, variant, CARRY_32};
      u32 exponent = primes.prevPrime(fft.maxExp());
      u32 best_zerohack_w = 0;
      double best_cost = -1.0;
      for (u32 zerohack_w : {0, 1}) {
        shared.args->flags["ZEROHACK_W"] = to_string(zerohack_w);
        double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
        log("Time for %12s using ZEROHACK_W=%u is %6.1f\n", fft.spec().c_str(), zerohack_w, cost);
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_zerohack_w = zerohack_w; }
      }
      log("Best ZEROHACK_W is %u.  Default ZEROHACK_W is 1.\n", best_zerohack_w);
      shared.args->flags["ZEROHACK_W"] = to_string(best_zerohack_w);
    }

    // Find best ZEROHACK_H setting
    if (1) {
      const FFTShape& shape = shapes[0];
      FFTConfig fft{shape, variant, CARRY_32};
      u32 exponent = primes.prevPrime(fft.maxExp());
      u32 best_zerohack_h = 0;
      double best_cost = -1.0;
      for (u32 zerohack_h : {0, 1}) {
        shared.args->flags["ZEROHACK_H"] = to_string(zerohack_h);
        double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
        log("Time for %12s using ZEROHACK_H=%u is %6.1f\n", fft.spec().c_str(), zerohack_h, cost);
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_zerohack_h = zerohack_h; }
      }
      log("Best ZEROHACK_H is %u.  Default ZEROHACK_H is 1.\n", best_zerohack_h);
      shared.args->flags["ZEROHACK_H"] = to_string(best_zerohack_h);
    }

    // Find best MIDDLE_IN_LDS_TRANSPOSE setting
    if (1) {
      const FFTShape& shape = shapes[0];
      FFTConfig fft{shape, variant, CARRY_32};
      u32 exponent = primes.prevPrime(fft.maxExp());
      u32 best_middle_in_lds_transpose = 0;
      double best_cost = -1.0;
      for (u32 middle_in_lds_transpose : {0, 1}) {
        shared.args->flags["MIDDLE_IN_LDS_TRANSPOSE"] = to_string(middle_in_lds_transpose);
        double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
        log("Time for %12s using MIDDLE_IN_LDS_TRANSPOSE=%u is %6.1f\n", fft.spec().c_str(), middle_in_lds_transpose, cost);
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_middle_in_lds_transpose = middle_in_lds_transpose; }
      }
      log("Best MIDDLE_IN_LDS_TRANSPOSE is %u.  Default MIDDLE_IN_LDS_TRANSPOSE is 1.\n", best_middle_in_lds_transpose);
      shared.args->flags["MIDDLE_IN_LDS_TRANSPOSE"] = to_string(best_middle_in_lds_transpose);
    }

    // Find best MIDDLE_OUT_LDS_TRANSPOSE setting
    if (1) {
      const FFTShape& shape = shapes[0];
      FFTConfig fft{shape, variant, CARRY_32};
      u32 exponent = primes.prevPrime(fft.maxExp());
      u32 best_middle_out_lds_transpose = 0;
      double best_cost = -1.0;
      for (u32 middle_out_lds_transpose : {0, 1}) {
        shared.args->flags["MIDDLE_OUT_LDS_TRANSPOSE"] = to_string(middle_out_lds_transpose);
        double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
        log("Time for %12s using MIDDLE_OUT_LDS_TRANSPOSE=%u is %6.1f\n", fft.spec().c_str(), middle_out_lds_transpose, cost);
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_middle_out_lds_transpose = middle_out_lds_transpose; }
      }
      log("Best MIDDLE_OUT_LDS_TRANSPOSE is %u.  Default MIDDLE_OUT_LDS_TRANSPOSE is 1.\n", best_middle_out_lds_transpose);
      shared.args->flags["MIDDLE_OUT_LDS_TRANSPOSE"] = to_string(best_middle_out_lds_transpose);
    }

    // Find best BIGLIT setting
    if (1) {
      const FFTShape& shape = shapes[0];
      FFTConfig fft{shape, variant, CARRY_32};
      u32 exponent = primes.prevPrime(fft.maxExp());
      u32 best_biglit = 0;
      double best_cost = -1.0;
      for (u32 biglit : {0, 1}) {
        shared.args->flags["BIGLIT"] = to_string(biglit);
        double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
        log("Time for %12s using BIGLIT=%u is %6.1f\n", fft.spec().c_str(), biglit, cost);
        if (best_cost < 0.0 || cost < best_cost) { best_cost = cost; best_biglit = biglit; }
      }
      log("Best BIGLIT is %u.  Default BIGLIT is 1.  The BIGLIT=0 option will probably be deprecated.\n", best_biglit);
      shared.args->flags["BIGLIT"] = to_string(best_biglit);
    }

    //GW:  Time some IN/OUT_WG/SIZEX combos?
  }

  // Flags that prune the amount of shapes and variants to time.
  // These should be computed automatically and saved in the tune.txt or config.txt file.
  // Tune.txt file should have a version number.

  // A command line option to run more combinations (higher number skips more combos)
  int skip_some_WH_variants = 1;                // 0 = skip nothing, 1 = skip slower widths/heights unless they have better Z, 2 = only run fastest widths/heights

  // The width = height = 512 FFT shape is so good, we probably don't need to time the width = 1024, height = 256 shape.
  bool skip_1K_256 = 1;

// make command line args for this? 
skip_some_WH_variants = 2;   // should default be 1??
skip_1K_256 = 0;

//GW:  Suggest tuning with TAIL_KERNELS=2 even if production runs use TAIL_KERNELS=3

  // For each width, time the 001, 101, and 201 variants to find the fastest width variant.
  // In an ideal world we'd use the -time feature and look at the kCarryFused timing.  Then we'd save this info in config.txt or tune.txt.
  map<int, u32> fastest_width_variants;

  // For each height, time the 100, 101, and 102 variants to find the fastest height variant.
  // In an ideal world we'd use the -time feature and look at the tailSquare timing.  Then we'd save this info in config.txt or tune.txt.
  map<int, u32> fastest_height_variants;

  vector<TuneEntry> results = TuneEntry::readTuneFile(*args);

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

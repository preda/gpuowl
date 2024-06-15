// Copyright (C) Mihai Preda

#include "tune.h"
#include "Args.h"
#include "FFTConfig.h"
#include "Gpu.h"
#include "GpuCommon.h"
#include "Primes.h"
#include "log.h"
#include "File.h"
#include "CycleFile.h"

#include <numeric>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>

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

    vector<string> options = split(val, ',');

    if (key == "fft") {
      vector<string> outOptions;
      for (string& s : options) {
        for(FFTShape& c : FFTShape::multiSpec(s)) {
          outOptions.push_back(c.spec());
        }
      }
      options = outOptions;
    }

    params.push_back({key, options});
  }
  return permute(params);
}

string toString(TuneConfig config) {
  string s{};
  for (const auto& [k, v] : config) { s += k + '=' + v + ','; }
  s.pop_back();
  return s;
}

} // namespace

u32 Tune::fftSize() {
  string spec = shared.args->fftSpec;
  assert(!spec.empty());
  return FFTShape::fromSpec(spec).fftSize();
}

u32 Tune::exponentForBpw(double bpw) {
  return primes.nearestPrime(fftSize() * bpw + 0.5);
}

double Tune::zForBpw(double bpw, FFTConfig fft) {
  u32 exponent = exponentForBpw(bpw);
  // auto [fft, fftConfig] = FFTConfig::bestFit(exponent, shared.args->fftSpec);
  // shared.args->setConfig(config);
  auto [ok, res, roeSq, roeMul] = Gpu::make(q, exponent, shared, fft, false)->measureROE(true);
  double z = roeSq.z();
  if (!ok) { log("Error at bpw %.2f (z %.2f) : %s\n", bpw, z, fft.spec().c_str()); }
  return z;
}

double solve(double A, double B, double C) {
  double delta = B * B - 4 * A * C;
  delta = max(delta, 0.0);

  return (-B - sqrt(delta)) / (2 * A);
}

// Quadratic least squares over 4 points at x-coords {-1, 0, 1, 2}
array<double, 3> quadApprox(array<double, 4> z) {
  array<double, 4> y;
  for (int i = 0; i < 4; ++i) { y[i] = log2(z[i]); }

  // Fit A*x^2 + B*x + C to the 4 points (-1, y[0]), (0, y[1]), (1, y[2]), (2, y[3])

  double c1 = y[0] + y[1] + y[2] + y[3];  // y1 + y2 + y3 + y4;
  // double c2 = -y1      + y3 + 2*y4;
  // double c3 =  y1      + y3 + 4*y4;

  double A = (y[0] - y[1] - y[2] + y[3]) / 4;              // (c3 - c2 - c1) / 4;
  double B = (3 * (y[3] - y[0]) + (y[2] - y[1])) / 10 - A; // (2c2 - c1 -10A) / 10;
  double C = (c1 - 2 * B - 6 * A) / 4;

  return {A, B, C};
}

array<double, 3> quadApprox(array<double, 5> z) {
  decltype(z) y;
  for (int i = 0; i < int(z.size()); ++i) { y[i] = log2(z[i]); }

  double c1 = y[0] + y[1] + y[2] + y[3] + y[4];
  double c2 = 2 * (y[4] - y[0]) + (y[3] - y[1]);
  double c3 = 4 * (y[4] + y[0]) + (y[3] + y[1]);

  double B = c2 / 10;
  double A = (c3 - 2 * c1) / 14;
  double C = c1 / 5 - 2 * A;

  return {A, B, C};
}

array<double, 3> quadApprox(array<double, 7> z) {
  decltype(z) y;
  for (int i = 0; i < int(z.size()); ++i) { y[i] = log2(z[i]); }

  double c1 = y[3] + (y[1] + y[5]) + (y[0] + y[6]) + (y[2] + y[4]);
  double c2 = 3 * (y[6] - y[0]) + 2 * (y[5] - y[1]) + (y[4] - y[2]);
  double c3 = 9 * (y[6] + y[0]) + 4 * (y[5] + y[1]) + (y[4] + y[2]);

  double B = c2 / 28;
  double A = (c3 - 4 * c1) / 84;
  double C = (c1 - 28 * A) / 7;

  return {A, B, C};
}

// Find x such that A*x^2 + B*x + C = log2(target)
double solveForTarget(array<double, 3> coefs, double target) {
  auto [A, B, C] = coefs;
  return solve(A, B, C - log2(target));
}

array<double, 3> Tune::maxBpw(FFTConfig fft) {
  const double STEP = 0.1;
  double bpw1 = 17.8 - log2(fftSize() / double(13 * 512 * 1024)) * 0.28;

  array<double, 7> z;
  for (int i = 0; i < int(z.size()); ++i) { z[i] = zForBpw(bpw1 + i * STEP, fft); }

  auto ABC = quadApprox(z);

  double AS = ABC[0] / (STEP * STEP);
  double BS = ABC[1] / STEP;

  double AA = AS;
  double k = 18 - (bpw1 + 3 * STEP);
  double BB = 2 * k * AS + BS;
  double CC = AS * k * k + BS * k + ABC[2];
  double bpw = solveForTarget({AA, BB, CC}, 28) + 18;

  log("%.3f | %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f | %f %f %f | %s\n",
      bpw, z[0], z[1], z[2], z[3], z[4], z[5], z[6],
      AA, BB, CC, fft.spec().c_str());

  return {AA, BB, CC};
}

void Tune::ztune() {
  File ztune = File::openAppend("ztune.txt");
  ztune.printf("#\n# %s\n#\n", shortTimeStr().c_str());

  Args *args = shared.args;
  assert(!args->hasFlag("CLEAN") && !args->hasFlag("TRIG_HI"));

  string ztuneStr = args->ztune;
  auto configs = ztuneStr.empty() ? FFTShape::genConfigs() : FFTShape::multiSpec(ztuneStr);
  for (FFTShape shape : configs) {
    string spec = shape.spec();
    // args->fftSpec = spec;
    ztune.printf("# %s\n", spec.c_str());

    for (u32 variant = 0; variant < FFTConfig::N_VARIANT; ++variant) {
      FFTConfig fft{shape, variant};
      auto [A, B, C] = maxBpw(fft);
      const double TARGET = 28;
      double bpw = 18 + solveForTarget({A, B, C}, TARGET);

      ztune.printf("{%f, %f, %f, \"%s\"}, // %.3f\n", A, B, C, fft.spec().c_str(), bpw);
    }
  }
}

bool shouldSkip(FFTConfig fft, double bestCaseCost, const vector<TuneEntry>& results) {
  // We assume that variant==0 is the fastest, so the cost at variant>0 will be larger.
  // If the current variant can't be worth it (relative to existing measurements), skip it.
  for (const TuneEntry& e : results) {
    if (e.cost <= bestCaseCost && e.fft.maxExp() >= fft.maxExp()) {
      log("skipping %s because of %s\n", fft.spec().c_str(), e.fft.spec().c_str());
      return true;
    }
  }
  return false;
}

pair<TuneConfig, double> Tune::findBestConfig(FFTConfig fft, const vector<TuneConfig>& configs) {
  u32 exponent = primes.prevPrime(fft.maxExp());

  // Every new FFTShape starts with variant zero, at which point we search for the best config for that
  // shape. The other variants of the same shape use the best config.
  TuneConfig bestConfig;
  double bestCost = 100; // arbitrary large value

  for (const auto& config : configs) {
    // assert(k == "IN_WG" || k == "OUT_WG" || k == "IN_SIZEX" || k == "OUT_SIZEX");
    shared.args->setConfig(config);
    // for (auto& [k, v] : config) { shared.args->flags[k] = v; }

    auto cost = Gpu::make(q, exponent, shared, fft, false)->timePRP();

    bool isBest = (cost < bestCost);
    if (isBest) {
      bestCost = cost;
      bestConfig = config;
    }
    log("%c %6.0f : %s %9u %s\n",
        isBest ? '*' : ' ', cost * 1e6, fft.spec().c_str(), exponent, toString(config).c_str());
  }
  return {bestConfig, bestCost};
}

void Tune::tune() {
  Args *args = shared.args;
  string fftSpec = args->fftSpec;

  auto configs = getTuneConfigs(shared.args->tune);
  if (configs.empty()) { configs.push_back({}); }

  vector<TuneEntry> results;

  for (const FFTShape& shape : FFTShape::multiSpec(args->fftSpec)) {
    FFTConfig zero{shape, 0};
    auto [config, costZero] = findBestConfig(zero, configs);
    string sconfig = toString(config);
    results.push_back({costZero, zero, sconfig});

    for (auto& [k, v] : config) { shared.args->flags[k] = v; }

    for (u32 variant = 1; variant < FFTConfig::N_VARIANT; ++variant) {
      FFTConfig fft{shape, variant};
      if (shouldSkip(fft, costZero, results)) { continue; }
      u32 exponent = primes.prevPrime(fft.maxExp());
      double cost = Gpu::make(q, exponent, shared, fft, false)->timePRP();
      log("  %6.0f : %s %9u %s\n", cost * 1e6, fft.spec().c_str(), exponent, sconfig.c_str());
      results.push_back({cost, fft, sconfig});
    }
  }

  log("\n");
  std::sort(results.begin(), results.end(),
            [](const TuneEntry& a, const TuneEntry& b) { return a.cost < b.cost; });

  CycleFile tuneFile{"tune.txt"};

  u32 prevMaxExp = 0;
  for (auto& [cost, fft, conf] : results) {
    tuneFile->printf("%6.0f %s %s\n", cost * 1e6, fft.spec().c_str(), conf.c_str());
    u32 maxExp = fft.maxExp();
    if (maxExp > prevMaxExp) {
      log("%6.0f %9u : %s %s\n", cost * 1e6, maxExp, fft.spec().c_str(), conf.c_str());
      prevMaxExp = maxExp;
    }
  }
}

SpeedConfig Speed::bestForExp(u32 exp) {
  // Ordered ascending by *cost*, the best is the first that's acceptable for *exp*.
  for (const SpeedConfig& c : configs) {
    if (c.maxExp >= exp) { return c; }
  }
  log("No acceptable FFT config found for exponent %u\n", exp);
  throw "No FFT";
}

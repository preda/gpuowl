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

double solve(double A, double B, double C) {
  double delta = B * B - 4 * A * C;
  delta = max(delta, 0.0);

  return (-B - sqrt(delta)) / (2 * A);
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
  for (const Entry& e : results) { s += formatEntry(e); }
  return s;
}

} // namespace

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


void Tune::ztune() {
  File ztune = File::openAppend("ztune.txt");
  ztune.printf("#\n# %s\n#\n", shortTimeStr().c_str());

  Args *args = shared.args;
  assert(!args->hasFlag("CLEAN") && !args->hasFlag("TRIG_HI"));

  string ztuneStr = args->fftSpec;
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

void Tune::ctune() {
  Args *args = shared.args;
  string fftSpec = args->fftSpec;

  auto configs = getTuneConfigs(shared.args->ctune);
  assert(!configs.empty());
  // if (configs.empty()) { configs.push_back({}); }


  vector<Entry> results;
  vector<Entry> secondResults;

  for (const FFTShape& shape : FFTShape::multiSpec(args->fftSpec)) {
    FFTConfig fft{shape, 0};
    u32 exponent = primes.prevPrime(fft.maxExp());
    log("tuning %10s with exponent %u\n", fft.shape.spec().c_str(), exponent);

    Entry best{{0, 0, 0}, {}, 1e9};
    Entry second{best};

    for (const auto& config : configs) {
      // assert(k == "IN_WG" || k == "OUT_WG" || k == "IN_SIZEX" || k == "OUT_SIZEX");
      shared.args->setConfig(config);

      auto cost = Gpu::make(q, exponent, shared, fft, false)->timePRP();

      bool isBest = (cost < best.cost);
      if (isBest) {
        second = best;
        best = {fft.shape, config, cost};
      }
      log("%c %6.0f : %s %s\n",
          isBest ? '*' : ' ', cost, fft.shape.spec().c_str(), toString(config).c_str());
    }
    results.push_back(best);
    secondResults.push_back(second);

    log("%s", formatEntry(best).c_str());
  }

  log("Second best configs (for information only):\n%s", formatConfigResults(secondResults).c_str());
  log("\nBest configs (lines can be copied to config.txt):\n%s", formatConfigResults(results).c_str());
}

void Tune::tune() {
  Args *args = shared.args;
  string fftSpec = args->fftSpec;

  vector<TuneEntry> results = TuneEntry::readTuneFile();

  for (const FFTShape& shape : FFTShape::multiSpec(args->fftSpec)) {
    double costZero{};
    for (u32 variant = 0; variant < FFTConfig::N_VARIANT; ++variant) {
      FFTConfig fft{shape, variant};

      if (variant > 0 && !TuneEntry{costZero, fft}.willUpdate(results)) {
        log("skipped %s\n", fft.spec().c_str());
        continue;
      }

      u32 maxExp = fft.maxExp();
      u32 exponent = primes.prevPrime(maxExp);
      double cost = Gpu::make(q, exponent, shared, fft, false)->timePRP();
      if (variant == 0) { costZero = cost; }

      bool isUseful = TuneEntry{cost, fft}.update(results);
      log("%c %6.0f %12s %9u\n", isUseful ? '*' : ' ', cost, fft.spec().c_str(), exponent);
    }
  }

  TuneEntry::writeTuneFile(results);
}

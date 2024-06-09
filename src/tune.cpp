// Copyright (C) Mihai Preda

#include "tune.h"
#include "Args.h"
#include "FFTConfig.h"
#include "Gpu.h"
#include "GpuCommon.h"
#include "Primes.h"
#include "log.h"
#include "File.h"

#include <numeric>
#include <string>
#include <vector>
#include <cinttypes>
#include <cassert>
#include <algorithm>

using std::accumulate;

using namespace std;

namespace {

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

using TuneConfig = vector<KeyVal>;

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

vector<TuneConfig> getRoeConfigs(const string& fftSpec) {
  // vector<FFTConfig> fftConfigs = FFTConfig::multiSpec(fftSpec);
  return getTuneConfigs("fft=" + fftSpec + ";CLEAN=0,1;TRIG_HI=0,1");
}

string toSimpleString(TuneConfig config) {
  assert(!config.empty() && config.front().first == "fft");
  string s = config.front().second;
  config.erase(config.begin());
  for (const auto& [k, v] : config) {
    assert(k != "fft" && k != "FFT");
    s += ","s + k + '=' + v;
  }
  return s;
}

string toString(TuneConfig config) {
  string s;

  if (config.empty()) { return s; }

  auto [k, v] = config.front();
  if (k == "fft") {
    s = "-fft " + v;
    config.erase(config.begin());
  }

  if (config.empty()) { return s; }

  s += s.empty() ? "-use " : " -use ";

  for (const auto& [k, v] : config) {
    assert(k != "fft" && k != "FFT" && k != "bpw");

    if (s.back() != ' ') { s += ','; }
    s += k + '=' + v;
  }

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

double Tune::zForBpw(double bpw, const string& config, Gpu* gpu) {
  auto [ok, res, roeSq, roeMul] = gpu->measureROE(shared.args->quickTune);
  double z = roeSq.z();
  if (!ok) { log("Error at bpw %.2f (z %.2f) : %s\n", bpw, z, config.c_str()); }
  return z;
}

double Tune::zForBpw(double bpw, const string& config) {
  u32 exponent = exponentForBpw(bpw);
  auto gpu = Gpu::make(q, exponent, shared, false);
  return zForBpw(bpw, config, gpu.get());
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

array<double, 3> Tune::maxBpw(const string& config) {
  const double STEP = 0.1;
  double bpw1 = 17.8 - log2(fftSize() / double(13 * 512 * 1024)) * 0.28;

  array<double, 7> z;
  for (int i = 0; i < int(z.size()); ++i) { z[i] = zForBpw(bpw1 + i * STEP, config); }

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
      AA, BB, CC, config.c_str());

  return {AA, BB, CC};

  /*
  // We deem z==TARGET safe-enough. And z==TARGET-5 an agressive lower limit.
  // dx becomes the BPW margin (the difference safe and agressive BPW)
  const double TARGET = 28;
  double xraw = solveForTarget(ABC, TARGET);
  // Scale xraw from [-2, 2] to [bpw2-STEP, bpw2+STEP]
  double bpw = bpw1 + (xraw + 3) * STEP;

  // double xrawLimit = solveForTarget(ABC, TARGET - 5);
  // double dx = (xrawLimit - xraw) * STEP;

  double x = bpw - 18;
  double x2 = solveForTarget({AA, BB, CC}, TARGET) + 18;

  log("%.3f %.3f %9u | %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f | %6.3f %6.3f %6.3f | %f %f %f %f | %s\n",
      bpw, x2, exponentForBpw(bpw),
      z[0], z[1], z[2], z[3], z[4], z[5], z[6],
      ABC[0], ABC[1], ABC[2], AA, BB, CC, AA * x * x + BB * x + CC,
      config.c_str());

  return {bpw, 0};
  */
}

void Tune::ztune() {
  auto configs = getRoeConfigs(shared.args->roeTune);
  File ztune = File::openAppend("ztune.txt");
  ztune.printf("#\n# %s\n#\n", shortTimeStr().c_str());
  string prevFftSpec{};

  for (const auto& config : configs) {
    for (auto& [k, v] : config) {
      if (k == "fft") {
        shared.args->fftSpec = v;
        if (prevFftSpec != v) {
          ztune.printf("# %s\n", v.c_str());
          prevFftSpec = v;
        }
      } else {
        shared.args->flags[k] = v;
      }
    }

    auto [A, B, C] = maxBpw(toString(config));
    const double TARGET = 28;
    double bpw = 18 + solveForTarget({A, B, C}, TARGET);

    ztune.printf("{%f, %f, %f, \"%s\"}, // %.3f\n", A, B, C, toSimpleString(config).c_str(), bpw);
  }
}

#if 0
void roeTune(Queue* q, GpuCommon shared) {
  auto configs = getTuneConfigs(shared.args->roeTune);
  Primes primes;

  u32 exponent = shared.args->prpExp;

  for (const auto& config : configs) {
    for (auto& [k, v] : config) {
      if (k == "fft") {
        shared.args->fftSpec = v;        
      } else if (k == "bpw") {
        double bpw = stod(v);
        exponent = primes.nearestPrime(FFTConfig::fromSpec(shared.args->fftSpec).fftSize() * bpw + 0.5);
      } else {
        shared.args->flags[k] = v;
      }
    }

    string fftSpec = shared.args->fftSpec;

    if (fftSpec.empty()) { throw "-roeTune without FFT spec"; }
    if (!exponent) { throw "-roeTune without exponent"; }

    u32 fftSize = FFTConfig::fromSpec(fftSpec).fftSize();

    auto gpu = Gpu::make(q, exponent, shared, false);
    auto [ok, res, roeSq, roeMul] = gpu->measureROE(shared.args->quickTune);

    log("%s %9d %016" PRIx64 " %.2f bpw %s %s %s\n",
        ok ? "OK" : "EE", exponent, res, exponent / double(fftSize),
        toString(config).c_str(), roeSq.toString().c_str(),
        shared.args->verbose ? roeMul.toString().c_str() : "");
  }
}
#endif

void tune(Queue* q, GpuCommon shared) {
  auto configs = getTuneConfigs(shared.args->tune);
  vector<pair<double, string>> results;

  if (!shared.args->prpExp) {
    log("-tune without any exponent being set. Use -prp <exponent> with -tune");
    throw "Use -prp <exponent> with -tune";
  }

  u32 exponent = shared.args->prpExp;

  double bestYet = 1000;

  for (const auto& config : configs) {
    // log("Timing %s\n", toString(config).c_str());
    for (auto& [k, v] : config) {
      if (k == "fft") {
        shared.args->fftSpec = v;
      } else {
        // assert(k == "IN_WG" || k == "OUT_WG" || k == "IN_SIZEX" || k == "OUT_SIZEX" || k == "OUT_SPACING");
        shared.args->flags[k] = v;
      }
    }
    if (!exponent) {
      log("No exponent in tune\n");
      throw "The exponent E=<N> must be set in tune=<values>";
    }
    auto gpu = Gpu::make(q, exponent, shared, false);
    auto [secsPerIt, res64] = gpu->timePRP(shared.args->quickTune);
    if (secsPerIt < 0) {
      log("Error %016" PRIx64 " %s\n", res64, toString(config).c_str());
    } else {
      bool isBest = (secsPerIt <= bestYet);
      if (isBest) { bestYet = secsPerIt; }
      log("%c %6.1f : %016" PRIx64 " %s\n", isBest ? '*' : ' ', secsPerIt * 1e6, res64, toString(config).c_str());
    }
    results.push_back({secsPerIt < 0 ? 1.0 : secsPerIt, toString(config)});
  }

  log("\n");
  std::sort(results.begin(), results.end());
  for (auto& [secs, conf] : results) {
    if (secs >= 1.0) {
      log("Error : %s\n", conf.c_str());
    } else {
      log("%6.1f : %s\n", secs * 1e6, conf.c_str());
    }
  }
}

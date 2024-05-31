// Copyright (C) Mihai Preda

#include "tune.h"
#include "Args.h"
#include "FFTConfig.h"
#include "Gpu.h"
#include "GpuCommon.h"
#include "Primes.h"
#include "log.h"
#include "File.h"

#include <string>
#include <vector>
#include <cinttypes>
#include <cassert>
#include <algorithm>

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
        for(FFTConfig& c : FFTConfig::multiSpec(s)) {
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
  return getTuneConfigs("fft=" + fftSpec + ";DIRTY=1,0;TRIG_HI=0,1");
}

string toSimpleString(TuneConfig config) {
  assert(!config.empty() && config.front().first == "fft");
  string s = config.front().second + ' ';
  config.erase(config.begin());
  for (const auto& [k, v] : config) {
    assert(k != "fft" && k != "FFT");

    if (s.back() != ' ') { s += ','; }
    s += k + '=' + v;
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
  return FFTConfig::fromSpec(spec).fftSize();
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

array<double, 3> interpolate4(const vector<double>& zv, double target) {
  assert(zv.size() == 4);
  vector<double> y;
  for (double z : zv) { y.push_back(log2(z)); }

  // Below we do a quadratic least-squares fit to the 4 points
  // (-1, z1), (0, z2), (1, z3), (2, z4)
  // with A*x^2 + B*x + C

  double c1 = y[0] + y[1] + y[2] + y[3];  // y1 + y2 + y3 + y4;
  // double c2 = -y1      + y3 + 2*y4;
  // double c3 =  y1      + y3 + 4*y4;

  double A = (y[0] - y[1] - y[2] + y[3]) / 4;            // (c3 - c2 - c1) / 4;
  double B = (3 * (y[3] - y[0]) + (y[2] - y[1])) / 10 - A; // (2c2 - c1 -10A) / 10;
  double C = (c1 - 2 * B - 6 * A) / 4;

  C -= log2(target);

  double x = solve(A, B, C);

  return {x, A, B};


}

pair<double, u32> Tune::maxBpw(const string& config) {
  const double STEP = 0.2;
  double bpw1 = 17.9 - log2(fftSize() / double(13 * 512 * 1024)) * 0.28;

  vector<double> z;
  for (int i = 0; i < 4; ++i) { z.push_back(zForBpw(bpw1 + i * STEP, config)); }

  // We deem this a safe-enough target
  const double TARGET = 28;
  auto [xraw, A, B] = interpolate4(z, TARGET);

  // Scale xraw from [-1, 1] to [bpw2-STEP, bpw2+STEP]
  double bpw = bpw1 + (xraw + 1) * STEP;
  u32 maxExp = exponentForBpw(bpw);

  log("%.2f %9u | %5.2f %5.2f %5.2f %5.2f | %6.3f %6.3f | %s\n",
      bpw, maxExp, z[0], z[1], z[2], z[3], A, B, config.c_str());

  return {bpw, maxExp};
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

    auto [bpw, maxExp] = maxBpw(toString(config));

    ztune.printf("%9u %s # %.2f\n", maxExp, toSimpleString(config).c_str(), bpw);
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

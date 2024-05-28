// Copyright (C) Mihai Preda

#include "tune.h"
#include "Args.h"
#include "FFTConfig.h"
#include "Gpu.h"
#include "GpuCommon.h"
#include "Primes.h"
#include "log.h"

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

using TuneConfig = vector<pair<string, string>>;

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

    if (key == "bpw") {
      vector<string> outOptions;
      for (string& s : options) {
        if (s.find(':') != string::npos) {
          vector<string> parts = split(s, ':');
          assert(parts.size() == 3);
          double begin = stod(parts.at(0));
          double end = stod(parts.at(1));
          double step = stod(parts.at(2));
          assert(begin < end && step > 0);
          for (double x = begin; x < end; x += step) {
            outOptions.push_back(to_string(x));
          }
        } else {
          outOptions.push_back(s);
        }
      }
      options = outOptions;
    }

    params.push_back({key, options});
  }

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

string toString(TuneConfig config) {
  string s;

  if (config.empty()) { return s; }

  auto [k, v] = config.front();
  if (k == "fft") {
    s = "-fft " + v;
    config.erase(config.begin());
  }

  if (!config.empty() && (config.front().first == "bpw")) { config.erase(config.begin()); }

  if (config.empty()) { return s; }

  s += s.empty() ? "-use " : " -use ";

  for (const auto& [k, v] : config) {
    assert(k != "fft" && k != "FFT" && k != "bpw");

    if (s.back() != ' ') { s += ','; }
    s += k + '=' + v;
  }

  return s;
}

struct Point {
  double bpw;
  double z;
};

[[maybe_unused]] double interpolate(Point p1, Point p2, double target) {
    // double lowBPW, double z1, double highBPW, double z2, double targetZ) {
  /*
  assert(lowBPW < highBPW);
  assert(z1 >= targetZ && targetZ >= z2);
  assert(z1 > z2 && z2 > 0);
  */

  assert(p1.z != p2.z);

  double log1 = log(p1.z);
  double log2 = log(p2.z);
  double logt = log(target);

  return (p1.bpw * (log2 - logt) + p2.bpw * (logt - log1)) / (log2 - log1);
}


} // namespace

u32 Tune::exponentForBpw(double bpw) {
  string spec = shared.args->fftSpec;
  assert(!spec.empty());
  u32 fftSize = FFTConfig::fromSpec(spec).fftSize();
  return primes.nearestPrime(fftSize * bpw + 0.5);
}

double Tune::zForBpw(double bpw, const string& config) {
  u32 exponent = exponentForBpw(bpw);
  auto gpu = Gpu::make(q, exponent, shared, false);
  auto [ok, res, roeSq, roeMul] = gpu->measureROE(shared.args->quickTune);
  double z = roeSq.z();
  if (!ok) { log("Error at bpw %.2f (z %.2f) : %s\n", bpw, z, config.c_str()); }
  return z;
}

// Given the 3 points: (-1, z1), (0, z2), (1, z3),
// interpolate (x, target) using a second-degree curve over the log of the z-values; return x.
double interpolateZ(double z1, double z2, double z3, double target, bool verbose) {
  double y1 = log2(z1);
  double y2 = log2(z2);
  double y3 = log2(z3);

  // Find the poly A*x^2 + B*X + C pasing through (-1, y1), (0, y2), (1, y3).
  double B = (y3 - y1) / 2;
  double C = y2;
  double A = (y1 + y3 - 2 * C) / 2;

  // Solve the second-degree eq. Ax^2 + Bx + C = log2(target)

  C -= log2(target);

  if (verbose) { log("%5.2f %5.2f %5.2f | %5.2f %5.2f %5.2f\n", z1, z2, z3, A, B, C); }

  // We expect A < 0
  if (A >= 0) {
    return -C/B;
  } else {
    double delta = B * B - 4 * A * C;
    delta = max(delta, 0.0);

    // the larger of the two solutions, given A negative
    return (-B - sqrt(delta)) / (2 * A);
  }
  assert(false); // unreachable
}

void Tune::maxBpw(const string& config, u32 fftSize) {
  const double STEP = 0.25;
  double bpw1 = 18.02 - log2(fftSize / double(13 * 512 * 1024)) * 0.28;

  double bpw2 = bpw1 + STEP;
  double bpw3 = bpw2 + STEP;

  double z1 = zForBpw(bpw1, config);
  double z2 = zForBpw(bpw2, config);
  double z3 = zForBpw(bpw3, config);

  // We assume z>=27 a safe-enough target
  const double TARGET = 27;
  double xraw = interpolateZ(z1, z2, z3, TARGET, shared.args->verbose);

  // Scale xraw from [-1, 1] to [bpw2-STEP, bpw2+STEP]
  double x = bpw2 + xraw * STEP;

  log("%5.2f : %s\n", x, config.c_str());
}

void Tune::roeSearch() {
  auto configs = getTuneConfigs(shared.args->roeTune);

  for (const auto& config : configs) {
    for (auto& [k, v] : config) {
      if (k == "fft") {
        shared.args->fftSpec = v;
      } else {
        shared.args->flags[k] = v;
      }
    }

    u32 fftSize = FFTConfig::fromSpec(shared.args->fftSpec).fftSize();
    maxBpw(toString(config), fftSize);
  }
}

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

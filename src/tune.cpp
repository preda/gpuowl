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

pair<u32, double> findMaxExponent(Queue* q, GpuCommon shared, double target = 27) {
  Primes primes;
  double bpw = 18.3; // Some starting point in the search
  double step = 0.5;
  bool prevIsGood = true;
  bool isFirst = true;
  double low = 0, high = 1000;

  string spec = shared.args->fftSpec;
  u32 fftSize = FFTConfig::fromSpec(spec).fftSize();
  while (true) {
    u32 exponent = primes.nearestPrime(fftSize * bpw + 0.5);
    auto gpu = Gpu::make(q, exponent, shared, false);
    auto [ok, res, roeSq, roeMul] = gpu->measureROE(shared.args->quickTune);
    double z = roeSq.z();

    log("%s %s %u bpw=%.2f z=%.1f\n", ok ? "OK" : "EE", spec.c_str(), exponent, exponent / double(fftSize), z);

    bool good = (z >= target);

    assert(ok || !good);

    if (abs(z - target) < 0.5 || (good && step < 0.02)) {
      return {exponent, z};
    }

    bool crossed = !isFirst && good != prevIsGood;
    if (good) {
      assert(bpw >= low);
      low = bpw;
    } else {
      assert(bpw <= high);
      high = bpw;
    }

    isFirst = false;
    prevIsGood = good;

    if (crossed) { step /= 2; }
    double next = bpw + (good ? step : -step);
    if (abs(next - low) < 0.001 || abs(high - next) < 0.001) { step /= 2; }
    bpw += good ? step : -step;
  }
  assert(false);
  return {0, 0};
}

}

void roeSearch(Queue* q, GpuCommon shared) {
  auto configs = getTuneConfigs(shared.args->roeTune);

  if (!shared.args->flags.contains("STATS")) { shared.args->flags["STATS"] = "15";}

  for (const auto& config : configs) {
    for (auto& [k, v] : config) {
      if (k == "fft") {
        shared.args->fftSpec = v;
      } else {
        shared.args->flags[k] = v;
      }
    }

    if (shared.args->fftSpec.empty()) { throw "-roeTune without FFT spec"; }
    u32 fftSize = FFTConfig::fromSpec(shared.args->fftSpec).fftSize();;
    auto [exponent, z] = findMaxExponent(q, shared);
    log("%u : BPW=%.2f Z=%.1f %s\n", exponent, exponent/double(fftSize), z, toString(config).c_str());
  }
}

void roeTune(Queue* q, GpuCommon shared) {
  auto configs = getTuneConfigs(shared.args->roeTune);
  Primes primes;

  if (!shared.args->flags.contains("STATS")) {
    shared.args->flags["STATS"] = "15";
  }

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
        toString(config).c_str(), roeSq.toString(0).c_str(),
        shared.args->verbose ? roeMul.toString(0).c_str() : "");
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
      log("%6.1f : %016" PRIx64 " %s\n", secsPerIt * 1e6, res64, toString(config).c_str());
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

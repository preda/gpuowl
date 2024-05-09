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
    auto [ok, res, roe] = gpu->measureROE();

    log("%s %9d %016" PRIx64 " %.2f bpw %s %s\n",
        ok ? "OK" : "EE", exponent, res, exponent / double(fftSize),
        toString(config).c_str(), roe.toString(0).c_str());
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

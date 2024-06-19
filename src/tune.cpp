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

pair<double, double> Tune::maxBpw(FFTConfig fft) {
  const double STEP = 0.06;

  double oldBpw = fft.maxBpw();
  double bpw = oldBpw;

  const double TARGET = 28;

  double z[4]{};

  z[1] = zForBpw(bpw - STEP, fft);
  if (z[1] < TARGET) {
    log("step down from %f\n", bpw);
    bpw -= 2*STEP;
    z[2] = z[1];
    z[1] = zForBpw(bpw - STEP, fft);
  } else {
    z[2] = zForBpw(bpw + STEP, fft);
    if (z[2] > TARGET) {
      log("step up from %f\n", bpw);
      bpw += 2*STEP;
      z[1] = z[2];
      z[2] = zForBpw(bpw + STEP, fft);
    }
  }
  z[0] = zForBpw(bpw - 2 * STEP, fft);
  z[3] = zForBpw(bpw + 2 * STEP, fft);

  double A = (2 * (z[3] - z[0]) + (z[2] - z[1])) / (10 * STEP);
  double B = (z[0] + z[1] + z[2] + z[3]) / 4;
  double x = bpw + (TARGET - B) / A;

  log("%s %.3f -> %.3f | %.2f %.2f %.2f %.2f | %.0f %.1f\n",
      fft.spec().c_str(), bpw, x, z[0], z[1], z[2], z[3], -A, B);
  return {x, -A};
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
  auto [ok, res, roeSq, roeMul] = Gpu::make(q, exponent, shared, fft, false)->measureROE(true);
  double z = roeSq.z();
  if (!ok) { log("Error at bpw %.2f (z %.2f) : %s\n", bpw, z, fft.spec().c_str()); }
  return z;
}

void Tune::ztune() {
  File ztune = File::openAppend("ztune.txt");
  ztune.printf("#\n# %s\n#\n", shortTimeStr().c_str());

  Args *args = shared.args;

  string ztuneStr = args->fftSpec;
  auto configs = ztuneStr.empty() ? FFTShape::genConfigs() : FFTShape::multiSpec(ztuneStr);
  for (FFTShape shape : configs) {
    string spec = shape.spec();
    // ztune.printf("# %s\n", spec.c_str());

    double bpw[4];
    double A[4];
    for (u32 variant = 0; variant < FFTConfig::N_VARIANT; ++variant) {
      FFTConfig fft{shape, variant};
      std::tie(bpw[variant], A[variant]) = maxBpw(fft);
    }
    string s = "\""s + spec + "\"";
    ztune.printf("{%12s, {%.3f, %.3f, %.3f, %.3f}, {%.0f, %.0f, %.0f, %.0f}},\n",
                 s.c_str(), bpw[0], bpw[1], bpw[2], bpw[3], A[0], A[1], A[2], A[3]);
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
      fft.config = config;

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

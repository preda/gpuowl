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

pair<double, double> linearFit(double* z, double STEP) {
  double A = (2 * (z[3] - z[0]) + (z[2] - z[1])) / (10 * STEP);
  double B = (z[0] + z[1] + z[2] + z[3]) / 4;
  return {A, B};
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

  // double A = (2 * (z[3] - z[0]) + (z[2] - z[1])) / (10 * STEP);
  // double B = (z[0] + z[1] + z[2] + z[3]) / 4;

  auto [A, B] = linearFit(z, STEP);

  double x = bpw + (TARGET - B) / A;

  log("%s %.3f -> %.3f | %.2f %.2f %.2f %.2f | %.0f %.1f\n",
      fft.spec().c_str(), bpw, x, z[0], z[1], z[2], z[3], -A, B);
  return {x, -A};
}

double Tune::zForBpw(double bpw, FFTConfig fft) {
  u32 exponent = primes.nearestPrime(fft.size() * bpw + 0.5);
  auto [ok, res, roeSq, roeMul] = Gpu::make(q, exponent, shared, fft, {}, false)->measureROE(true);
  double z = roeSq.z();
  if (!ok) { log("Error at bpw %.2f (z %.2f) : %s\n", bpw, z, fft.spec().c_str()); }
  return z;
}

void Tune::ztune() {
  File ztune = File::openAppend("ztune.txt");
  ztune.printf("\n// %s\n\n", shortTimeStr().c_str());
  auto configs = FFTShape::multiSpec(shared.args->fftSpec);
  for (FFTShape shape : configs) {
    double bpw[4];
    double A[4];
    for (u32 variant = 0; variant < FFTConfig::N_VARIANT; ++variant) {
      FFTConfig fft{shape, variant};
      std::tie(bpw[variant], A[variant]) = maxBpw(fft);
    }
    string s = "\""s + shape.spec() + "\"";
    ztune.printf("{%12s, {%.3f, %.3f, %.3f, %.3f}, {%.0f, %.0f, %.0f, %.0f}},\n",
                 s.c_str(), bpw[0], bpw[1], bpw[2], bpw[3], A[0], A[1], A[2], A[3]);
  }
}

void Tune::carryTune() {
  File fo = File::openAppend("carrytune.txt");
  fo.printf("\n// %s\n\n", shortTimeStr().c_str());
  shared.args->flags["STATS"] = "1";
  u32 prevSize = 0;
  for (FFTShape shape : FFTShape::multiSpec(shared.args->fftSpec)) {
    FFTConfig fft{shape, 3};
    if (prevSize == fft.size()) { continue; }
    prevSize = fft.size();

    vector<double> zv;
    const double mid = fft.shape.carryLimitBPW();
    for (double bpw : {mid - 0.05, mid + 0.05}) {
      u32 exponent = primes.nearestPrime(fft.size() * bpw);
      auto [ok, carry] = Gpu::make(q, exponent, shared, fft, {}, false)->measureCarry();
      if (!ok) { log("Error %s at %f\n", fft.spec().c_str(), bpw); }
      zv.push_back(carry.z());
    }

    double avg = (zv[0] + zv[1]) / 2;
    log("%14s %.3f : %.3f (%.3f %.3f)\n", fft.spec().c_str(), mid, avg, zv[0], zv[1]);
    fo.printf("%f %f\n", log2(fft.size()), avg);

#if 0
    const double STEP = 0.06;
    // const double INI_BPW = fft.maxBpw() + STEP;
    double mid = fft.maxBpw() - STEP;
    string spec = fft.spec();

    vector<double> zv;
    for (double bpw : {mid - 2*STEP, mid - STEP, mid + STEP, mid + 2*STEP}) {
      u32 exponent = primes.nearestPrime(fft.size() * bpw);
      auto [ok, carry] = Gpu::make(q, exponent, shared, fft, {}, false)->measureCarry();
      if (!ok) { log("Error %s at %f\n", spec.c_str(), bpw); }
      double z = carry.z();
      log("%14s %u %.3f : %u %.3f %f\n", spec.c_str(), exponent, bpw, carry.N, z, carry.max);
      zv.push_back(z);
    }
    auto [A, B] = linearFit(zv.data(), STEP);
    log("%12s %.3f : %f %f\n", spec.c_str(), mid, A, B);
    fo.printf("%f %.3f %f %f\n", log2(fft.size()), mid, A, B);
#endif
  }
}

template<typename T>
void add(vector<T>& a, const vector<T>& b) {
  a.insert(a.end(), b.begin(), b.end());
}

void Tune::ctune() {
  Args *args = shared.args;

  vector<vector<TuneConfig>> configsVect;
  for (const string& s : shared.args->ctune) {
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
    u32 exponent = primes.prevPrime(FFTConfig{shape, 0}.maxExp());
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
        auto cost = Gpu::make(q, exponent, shared, FFTConfig{shape, 0}, c, false)->timePRP();

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

  vector<TuneEntry> results = TuneEntry::readTuneFile(*args);

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
      double cost = Gpu::make(q, exponent, shared, fft, {}, false)->timePRP();
      if (variant == 0) { costZero = cost; }

      bool isUseful = TuneEntry{cost, fft}.update(results);
      log("%c %6.0f %12s %9u\n", isUseful ? '*' : ' ', cost, fft.spec().c_str(), exponent);
    }
  }

  TuneEntry::writeTuneFile(results);
}

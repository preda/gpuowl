// Copyright (C) Mihai Preda

#pragma once

#include "Primes.h"
#include "GpuCommon.h"
#include "FFTConfig.h"

#include <array>
#include <vector>

class Queue;
class GpuCommon;
class RoeInfo;
class Gpu;

struct SpeedConfig {
  u32 maxExp;
  double cost;

  FFTConfig fft;

  string config; // vector<KeyVal>
};

class Speed {
  vector<SpeedConfig> configs; // ordered ascending on "cost"

public:
  SpeedConfig bestForExp(u32 exp);

};

using TuneConfig = vector<KeyVal>;

class Tune {
private:
  u32 fftSize();
  std::array<double, 3> maxBpw(FFTConfig fft);

  std::pair<TuneConfig, double> findBestConfig(FFTConfig, const vector<TuneConfig>& configs);

public:
  Queue *q;
  GpuCommon shared;

  Primes primes;

  u32 exponentForBpw(double bpw);
  double zForBpw(double bpw, FFTConfig fft);

  void ztune();
  void tune();
};

void roeTune(Queue* q, GpuCommon shared);
void roeSearch(Queue* q, GpuCommon shared);

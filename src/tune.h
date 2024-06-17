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

using TuneConfig = vector<KeyVal>;

class Tune {
private:
  Queue *q;
  GpuCommon shared;
  Primes primes;

  u32 fftSize();
  std::array<double, 3> maxBpw(FFTConfig fft);
  u32 exponentForBpw(double bpw);
  double zForBpw(double bpw, FFTConfig fft);

public:
  Tune(Queue *q, GpuCommon shared) : q{q}, shared{shared} {}

  // Find the max-BPW for each FFT
  void ztune();

  // Find the best configuration for each FFT
  void ctune();

  // Considering the cost of each FFT and the max-BPW, work out the transition points between them
  void tune();
};

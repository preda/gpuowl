// Copyright (C) Mihai Preda

#pragma once

#include "Primes.h"
#include "GpuCommon.h"

#include <array>

class Queue;
class GpuCommon;
class RoeInfo;
class Gpu;

class Tune {
private:
  u32 fftSize();
  std::array<double, 3> maxBpw(const std::string& config);

public:
  Queue *q;
  GpuCommon shared;

  Primes primes;

  u32 exponentForBpw(double bpw);
  double zForBpw(double bpw, const string& config);
  double zForBpw(double bpw, const string& config, Gpu* gpu);


  void ztune();
};

void tune(Queue* q, GpuCommon shared);
void roeTune(Queue* q, GpuCommon shared);
void roeSearch(Queue* q, GpuCommon shared);

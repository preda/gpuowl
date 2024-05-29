// Copyright (C) Mihai Preda

#pragma once

#include "Primes.h"
#include "GpuCommon.h"

class Queue;
class GpuCommon;
class RoeInfo;
class Gpu;

struct TuneEntry {
  double secsPerIt;
  u32 maxExp;
  std::string config;
  u32 fftSize;

  std::string toString() const;
};

class Tune {
public:
  Queue *q;
  GpuCommon shared;

  Primes primes;

  u32 exponentForBpw(double bpw);
  double zForBpw(double bpw, const string& config);
  double zForBpw(double bpw, const string& config, Gpu* gpu);

  TuneEntry maxBpw(const std::string& config, u32 fftSize);

  void roeSearch();
};

void tune(Queue* q, GpuCommon shared);
void roeTune(Queue* q, GpuCommon shared);
void roeSearch(Queue* q, GpuCommon shared);

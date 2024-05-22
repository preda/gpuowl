// Copyright (C) Mihai Preda

#pragma once

#include "Primes.h"
#include "GpuCommon.h"

class Queue;
class GpuCommon;
class RoeInfo;

class Tune {
public:
  Queue *q;
  GpuCommon shared;

  Primes primes;

  u32 exponentForBpw(double bpw);
  RoeInfo zForBpw(double bpw);

  void maxBpw(const std::string& config, u32 fftSize);

  void roeSearch();
};

void tune(Queue* q, GpuCommon shared);
void roeTune(Queue* q, GpuCommon shared);
void roeSearch(Queue* q, GpuCommon shared);

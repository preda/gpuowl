// Copyright (C) Mihai Preda

#pragma once

#include "FFTConfig.h"

#include <vector>

class Args;

class TuneEntry {
public:
  double cost;
  FFTConfig fft;

  bool update(std::vector<TuneEntry>&) const;
  bool willUpdate(const vector<TuneEntry>&) const;

  static vector<TuneEntry> readTuneFile(const Args& args);
  static void writeTuneFile(const vector<TuneEntry>&);
};

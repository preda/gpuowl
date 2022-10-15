// Copyright (C) 2017-2022 Mihai Preda.

#pragma once

#include "common.h"

#include <chrono>
#include <string>

class Timer {
  using clock = std::chrono::steady_clock;
  clock::time_point start;

public:
  Timer() : start(clock::now()) {}
  

  double at() const { return std::chrono::duration<double>(clock::now() - start).count(); }

  double reset() {
    auto now = clock::now();
    double ret = std::chrono::duration<double>(now - start).count();
    start = now;
    return ret;
  }

  // void reset() { start = clock::now(); }

  /*
  u64 deltaNanos() {
    auto now = clock::now();
    u64 ret = std::chrono::duration<u64, std::nano>(now - start).count();
    start = now;
    return ret;
  }
  */
};

std::string timeStr();
std::string timeStr(const char *format);

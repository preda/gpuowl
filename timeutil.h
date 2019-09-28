// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include <chrono>
#include <string>

class Timer {
  using clock = std::chrono::high_resolution_clock;
  clock::time_point start;

public:
  Timer() : start(clock::now()) {}
  
  void reset() { start = clock::now(); }

  float elapsed() const { return std::chrono::duration<float>(clock::now() - start).count(); }

  float delta() {
    auto now = clock::now();
    double ret = std::chrono::duration<float>(now - start).count();
    start = now;
    return ret;
  }
};

std::string timeStr();
std::string timeStr(const char *format);

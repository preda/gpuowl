// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include <chrono>
#include <string>

using namespace std::chrono;

class Timer {
  high_resolution_clock::time_point start;

public:
  Timer() : start(high_resolution_clock::now()) {}
  
  void reset() { start = high_resolution_clock::now(); }

  float elapsed() const { return duration<float>(high_resolution_clock::now() - start).count(); }

  float delta() {
    auto now = high_resolution_clock::now();
    double ret = duration<float>(now - start).count();
    start = now;
    return ret;
  }
};

std::string timeStr();
std::string timeStr(const char *format);

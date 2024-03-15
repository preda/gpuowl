// Copyright (C) 2017-2022 Mihai Preda.

#pragma once

#include "common.h"

#include <chrono>
#include <string>
#include <thread>

class Timer {
  using clock = std::chrono::steady_clock;
  clock::time_point start;

public:
  static void usleep(u32 us) { std::this_thread::sleep_for(std::chrono::microseconds(us)); }

  Timer() : start(clock::now()) {}
  

  double at() const { return std::chrono::duration<double>(clock::now() - start).count(); }

  double reset() {
    auto now = clock::now();
    double ret = std::chrono::duration<double>(now - start).count();
    start = now;
    return ret;
  }
};

std::string timeStr();
std::string timeStr(const char *format);

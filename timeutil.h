// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include <chrono>
#include <string>

using namespace std::chrono;

class Timer {
  high_resolution_clock::time_point prev;

  auto delta() {
    auto save = prev;
    return (prev = high_resolution_clock::now()) - save;
  }

  auto elapsed() const { return high_resolution_clock::now() - prev; }
  
public:
  Timer() : prev(high_resolution_clock::now()) { }
  
  long deltaMicros() { return duration_cast<microseconds>(delta()).count(); }
  int deltaMillis()  { return duration_cast<milliseconds>(delta()).count(); }
  float deltaSecs()  { return deltaMillis() * 0.001f; }

  int elapsedMillis() const { return duration_cast<milliseconds>(elapsed()).count(); }
  void reset() { deltaMillis(); }
};

std::string timeStr();
std::string timeStr(const char *format);

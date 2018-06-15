// Copyright (C) 2017 Mihai Preda.

#pragma once

#include <sys/time.h>

typedef unsigned long long u64;

class Timer {
  u64 prev;

  static u64 timeMicros() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return u64(tv.tv_sec) * 1000000 + tv.tv_usec;
  }

 public:
  Timer() : prev(timeMicros()) { }

  u64 deltaMicros() {
    u64 now = timeMicros();
    u64 delta = now - prev;
    prev = now;
    return delta;
  }

  int deltaMillis() { return (int) (deltaMicros() / 1000); }
};

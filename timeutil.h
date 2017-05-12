// Copyright (C) 2017 Mihai Preda.

#include <sys/time.h>

typedef unsigned char byte;
typedef long long i64;
typedef unsigned long long u64;

u64 timeMillis() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

u64 timeMicros() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_usec;
}

class Timer {
  u64 prev;
  
 public:
  Timer() : prev (timeMillis()) { }

  int delta() {
    u64 now = timeMillis();
    int d   = (int) (now - prev);
    prev    = now;
    return d;
  }
};

class MicroTimer {
  u64 prev;
  
 public:
  MicroTimer() : prev(timeMicros()) { }

  u64 delta() {
    u64 now = timeMicros();
    u64 d = (now > prev) ? now - prev : (1000000 + now - prev);
    prev = now;
    return d;
  }
};

class TimeCounter {
  MicroTimer *timer;
  u64 us;
  
 public:
  TimeCounter(MicroTimer *t) : timer(t) , us(0) { }

  void tick() { us += timer->delta(); }
  u64 get() { return us; }
  void reset() { us = 0; }
};

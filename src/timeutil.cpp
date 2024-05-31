// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "timeutil.h"

#include <ctime>

std::string timeStr(const char *format) {
  time_t t = time(NULL);
  char buf[64];
  strftime(buf, sizeof(buf), format, localtime(&t));
  return buf;
}

std::string timeStr() {
  time_t t = time(NULL);
  char buf[64];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", gmtime(&t));   // equivalent to: "%F %T"
  return buf;
}

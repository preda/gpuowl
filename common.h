#pragma once

#include <cstdio>
#include <cstdarg>
#include <memory>
#include <vector>

typedef unsigned char byte;
typedef long long i64;
typedef unsigned long long u64;
typedef int      i32;
typedef unsigned u32;
typedef unsigned __int128 u128;

static_assert(sizeof(u32) == 4,   "size u32");
static_assert(sizeof(u64) == 8,   "size u64");

namespace std {
  template<> struct default_delete<FILE> {
    void operator()(FILE *f) { if (f != nullptr) { fclose(f); } }
  };
}

#ifdef __GNUC__
void log(const char *fmt, ...) __attribute__ ((format(printf, 1, 2)));
#else
void log(const char *fmt, ...);
#endif

using namespace std; // std::string, std::pair, std::vector, std::unique_ptr;

vector<unique_ptr<FILE>> logFiles;

void log(const char *fmt, ...) {
  va_list va;
  for (auto &f : logFiles) {
    va_start(va, fmt);
    vfprintf(f.get(), fmt, va);
    va_end(va);
#if !(defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE))
    fflush(f.get());
#endif
  }
}

#ifndef DUAL
#define DUAL
#endif

// The git revision should be passed through -D on the compiler command line (see Makefile).
#ifndef REV
#define REV
#endif

#define VERSION "2.3-" REV

std::unique_ptr<FILE> open(const std::string &name, const char *mode, bool doLog = true) {
  std::unique_ptr<FILE> f{fopen(name.c_str(), mode)};
  if (!f && doLog) { log("Can't open '%s' (mode '%s')\n", name.c_str(), mode); }
  return f;
}

string timeStr(const char *format) {
  time_t t = time(NULL);
  char buf[64];
  strftime(buf, sizeof(buf), format, localtime(&t));
  return buf;
}

string timeStr() {
  time_t t = time(NULL);
  char buf[64];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S UTC", gmtime(&t));   // equivalent to: "%F %T"
  return buf;
}

string longTimeStr()  { return timeStr("%Y-%m-%d %H:%M:%S %Z"); }
string shortTimeStr() { return timeStr("%Y-%m-%d %H:%M:%S"); }

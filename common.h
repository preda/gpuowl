#pragma once

#include <cstdio>
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

void initLog();

using namespace std; // std::string, std::pair, std::vector, std::unique_ptr;

#ifndef DUAL
#define DUAL
#endif

// The git revision should be passed through -D on the compiler command line (see Makefile).
#ifndef REV
#define REV
#endif

#define VERSION "2.3-" REV

unique_ptr<FILE> open(const string &name, const char *mode, bool doLog = true);

string timeStr(const char *format);
string timeStr();
string longTimeStr();
string shortTimeStr();

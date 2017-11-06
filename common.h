#pragma once

#include <cstdio>
#include <memory>

typedef unsigned char byte;
typedef long long i64;
typedef unsigned long long u64;
typedef int      i32;
typedef unsigned u32;

typedef unsigned int uint;
typedef unsigned long ulong;

static_assert(sizeof(u32) == 4,   "size u32");
static_assert(sizeof(u64) == 8,   "size u64");
static_assert(sizeof(ulong) == 8, "size ulong");

#ifdef __GNUC__
void log(const char *fmt, ...) __attribute__ ((format(printf, 1, 2)));
#else
void log(const char *fmt, ...);
#endif

namespace std {
template<> struct default_delete<FILE> {
  void operator()(FILE *f) {
    // fprintf(stderr, "file closed\n");
    if (f != nullptr) { fclose(f); }
  }
 };
}

std::unique_ptr<FILE> open(const char *name, const char *mode, bool doLog = true) {
  std::unique_ptr<FILE> f{fopen(name, mode)};
  if (!f && doLog) { log("Can't open '%s' (mode '%s')\n", name, mode); }
  return f;
}

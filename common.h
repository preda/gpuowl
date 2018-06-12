#pragma once

#include <cstdio>
#include <memory>

typedef unsigned char byte;
typedef long long i64;
typedef unsigned long long u64;
typedef int      i32;
typedef unsigned u32;
typedef unsigned __int128 u128;

// OpenCL type names. Conflicts with <sys/types.h>.
// typedef u32 uint;
// typedef u64 ulong;

static_assert(sizeof(u32) == 4,   "size u32");
static_assert(sizeof(u64) == 8,   "size u64");

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

std::unique_ptr<FILE> open(const std::string &name, const char *mode, bool doLog = true) {
  std::unique_ptr<FILE> f{fopen(name.c_str(), mode)};
  if (!f && doLog) { log("Can't open '%s' (mode '%s')\n", name.c_str(), mode); }
  return f;
}

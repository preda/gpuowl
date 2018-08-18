#pragma once

#include <cstdio>
#include <cstdint>

#include <memory>
#include <vector>

typedef unsigned char byte;
typedef long long i64;
typedef unsigned long long u64;
typedef int      i32;
typedef unsigned u32;

// try one of these for a 128bit integer.
typedef unsigned __int128 u128;
// typedef uint128_t u128;
// typedef __uint128_t u128;

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

void initLog(const char *);

using namespace std; // std::string, std::pair, std::vector, std::unique_ptr;

// The git revision should be passed through -D on the compiler command line (see Makefile).
#ifndef REV
#define REV
#endif

#define VERSION "3.8-" REV
// Reason for version change:
// 3.8 : TF multi-bit in one pass. OWL TF savefile v2.
// 3.7 : integrated TF
// 3.6 : initial TF
// 3.5 : require OpenCL 2.x
// 3.4 : add factor-9 FFT step, and more FFT sizes.
// 3.3 : add new factor-5 FFT.
// 3.2 : always use fused tail. Drop un-fused fftH, square, multiply kernels.
// 3.1 : use computed trig (instead of tables) in transpose and tailFused / square. Reworked tailFused.
// 3.0 : add CUDA backend.

unique_ptr<FILE> open(const string &name, const char *mode, bool doLog = true);

string timeStr();
// string longTimeStr();

// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

typedef unsigned char byte;
typedef long long i64;
typedef unsigned long long u64;
typedef int      i32;
typedef unsigned u32;

static_assert(sizeof(u32) == 4,   "size u32");
static_assert(sizeof(u64) == 8,   "size u64");

#ifdef __GNUC__
void log(const char *fmt, ...) __attribute__ ((format(printf, 1, 2)));
#else
void log(const char *fmt, ...);
#endif

void initLog(const char *);

using namespace std; // std::string, std::pair, std::vector, std::unique_ptr;

#define BASE_VERSION "4.1"
// Reason for version change:
// 4.1 : introduce PRP-1
// 4.0 : add P-1 first stage; require GMP.
// 3.9 : use save/load path vs. commit/rollback
// 3.8 : TF multi-bit in one pass. OWL TF savefile v2.
// 3.7 : integrated TF
// 3.6 : initial TF
// 3.5 : require OpenCL 2.x
// 3.4 : add factor-9 FFT step, and more FFT sizes.
// 3.3 : add new factor-5 FFT.
// 3.2 : always use fused tail. Drop un-fused fftH, square, multiply kernels.
// 3.1 : use computed trig (instead of tables) in transpose and tailFused / square. Reworked tailFused.
// 3.0 : add CUDA backend.

// The git revision should be passed through -D on the compiler command line (see Makefile).
#ifdef REV
#define VERSION BASE_VERSION "-" REV
#else
#define VERSION BASE_VERSION
#endif

#define PROGRAM "gpuowl"

// string timeStr();

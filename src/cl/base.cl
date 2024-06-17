// Copyright (C) Mihai Preda and George Woltman

#pragma once

/* Tunable paramaters for -ctune :

IN_WG, OUT_WG: 64, 128, 256. Default: 256.
IN_SIZEX, OUT_SIZEX: 4, 8, 16, 32. Default: 32 on AMD, 4 on Nvidia.
UNROLL_W: 0, 1. Default: 0 on AMD, 1 on Nvidia.
UNROLL_H: 0, 1. Default: 1.
*/

/* List of code-specific macros. These are set by the C++ host code or derived
EXP        the exponent
WIDTH
SMALL_HEIGHT
MIDDLE
CARRY_LEN
NW
NH
AMDGPU  : if this is an AMD GPU
HAS_ASM : set if we believe __asm() can be used

-- Derived from above:
BIG_HEIGHT == SMALL_HEIGHT * MIDDLE
ND         number of dwords == WIDTH * MIDDLE * SMALL_HEIGHT
NWORDS     number of words  == ND * 2
G_W        "group width"  == WIDTH / NW
G_H        "group height" == SMALL_HEIGHT / NH
 */

// TRIG_TAB defaults to 0

#define STR(x) XSTR(x)
#define XSTR(x) #x

#define OVERLOAD __attribute__((overloadable))

#pragma OPENCL FP_CONTRACT ON

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef cl_khr_subgroups
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

// 64-bit atomics are not used ATM
// #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#if DEBUG
#define assert(condition) if (!(condition)) { printf("assert(%s) failed at line %d\n", STR(condition), __LINE__ - 1); }
// __builtin_trap();
#else
#define assert(condition)
//__builtin_assume(condition)
#endif // DEBUG

#if AMDGPU
// On AMDGPU the default is HAS_ASM
#if !NO_ASM
#define HAS_ASM 1
#endif
#endif // AMDGPU

// On Nvidia we need the old sync between groups in carryFused
#if !defined(OLD_FENCE) && !AMDGPU
#define OLD_FENCE 1
#endif

#if FFT_VARIANT > 3
#error FFT_VARIANT must be between 0 and 3
#endif

#define TRIG_HI (FFT_VARIANT & 1)
#define CLEAN (FFT_VARIANT >> 1)

#if CARRY32 && CARRY64
#error Conflict: both CARRY32 and CARRY64 requested
#endif

#if !CARRY32 && !CARRY64
// Presumably the carry should behave the same on AMD and Nvidia.
#define CARRY32 1
#endif

#if !defined(UNROLL_W)
#if AMDGPU
#define UNROLL_W 0
#else
#define UNROLL_W 1
#endif
#endif

#if !defined(UNROLL_H)
#define UNROLL_H 1
#endif

// Expected defines: EXP the exponent.
// WIDTH, SMALL_HEIGHT, MIDDLE.

#define BIG_HEIGHT (SMALL_HEIGHT * MIDDLE)
#define ND (WIDTH * BIG_HEIGHT)
#define NWORDS (ND * 2u)

#if (NW != 4 && NW != 8) || (NH != 4 && NH != 8)
#error NW and NH must be passed in, expected value 4 or 8.
#endif

#define G_W (WIDTH / NW)
#define G_H (SMALL_HEIGHT / NH)

/*
#if UNROLL_WIDTH
#define UNROLL_WIDTH_CONTROL
#else
#define UNROLL_WIDTH_CONTROL       __attribute__((opencl_unroll_hint(1)))
#endif
*/

typedef int i32;
typedef uint u32;
typedef long i64;
typedef ulong u64;

typedef i32 Word;
typedef int2 Word2;
typedef i64 CarryABM;

typedef double T;
typedef double2 T2;

#define RE(a) (a.x)
#define IM(a) (a.y)

#define P(x) global x * restrict
#define CP(x) const P(x)

#if AMDGPU
typedef constant const T2* Trig;
typedef constant const double2* BigTab;
#else
typedef global const T2* Trig;
typedef global const double2* BigTab;
#endif

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

void read(u32 WG, u32 N, T2 *u, const global T2 *in, u32 base) {
  for (i32 i = 0; i < N; ++i) { u[i] = in[base + i * WG + (u32) get_local_id(0)]; }
}

void write(u32 WG, u32 N, T2 *u, global T2 *out, u32 base) {
  for (i32 i = 0; i < N; ++i) { out[base + i * WG + (u32) get_local_id(0)] = u[i]; }
}

void bar() {
  // barrier(CLK_LOCAL_MEM_FENCE) is correct, but it turns out that on some GPUs
  // (in particular on Radeon VII and Radeon PRO VII) barrier(0) works as well and is faster.
  // So allow selecting the faster path when it works with -use FAST_BARRIER
#if FAST_BARRIER
  barrier(0);
#else
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
}

// On "classic" AMD GCN GPUs such as Radeon VII, the wavefront size was always 64. On RDNA GPUs the wavefront can
// be configured to be either 64 or 32. We use the FAST_BARRIER define as an indicator for GCN GPUs.
// On Nvidia GPUs the wavefront size is 32.
#if !WAVEFRONT
#if FAST_BARRIER && AMDGPU
#define WAVEFRONT 64
#else
#define WAVEFRONT 32
#endif
#endif

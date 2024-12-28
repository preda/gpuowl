// Copyright (C) Mihai Preda and George Woltman

#pragma once

/* Tunable paramaters for -ctune :

IN_WG, OUT_WG: 64, 128, 256. Default: 128.
IN_SIZEX, OUT_SIZEX: 4, 8, 16, 32. Default: 16.
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

// Nonteporal reads and writes might be a little bit faster on many GPUs by keeping more reusable data in the caches.
// However, on those GPUs with large caches there should be a significant speed gain from keeping FFT data in the caches.
// Default to the big win when caching is beneficial rather than the tiny gain when non-temporal is better.
#if !defined(NONTEMPORAL)
#define NONTEMPORAL 0
#endif

#if FFT_VARIANT > 3
#error FFT_VARIANT must be between 0 and 3
#endif

#if defined(TRIG_HI) || defined(CLEAN)
#error Use FFT_VARIANT instead of TRIG_HI or CLEAN
#endif

#define TRIG_HI (FFT_VARIANT & 1)
#define CLEAN (FFT_VARIANT >> 1)

#if !defined(UNROLL_W)
#if AMDGPU
#define UNROLL_W 0
#else
#define UNROLL_W 1
#endif
#endif

#if !defined(UNROLL_H)
#if AMDGPU && (SMALL_HEIGHT >= 1024)
#define UNROLL_H 0
#else
#define UNROLL_H 1
#endif
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

typedef int i32;
typedef uint u32;
typedef long i64;
typedef ulong u64;

typedef i32 Word;
typedef int2 Word2;

typedef double T;
typedef double2 T2;

#define RE(a) (a.x)
#define IM(a) (a.y)

#define P(x) global x * restrict
#define CP(x) const P(x)

// Macros for non-temporal load and store (in case we later want to provide a -use option to turn this off)
#if NONTEMPORAL && defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load) && __has_builtin(__builtin_nontemporal_store)
#define NTLOAD(mem)        __builtin_nontemporal_load(&(mem))
#define NTSTORE(mem,val)   __builtin_nontemporal_store(val, &(mem))
#else
#define NTLOAD(mem)        (mem)
#define NTSTORE(mem,val)   (mem) = val
#endif

// For reasons unknown, loading trig values into nVidia's constant cache has terrible performance
#if AMDGPU
typedef constant const T2* Trig;
#else
typedef global const T2* Trig;
#endif
// However, caching weights in nVidia's constant cache improves performance.
// Even better is to not pollute the constant cache with weights that are used only once.
// This requires two typedefs depending on how we want to use the BigTab pointer.
// For AMD we can declare BigTab as constant or global - it doesn't really matter.
typedef constant const double2* ConstBigTab;
#if AMDGPU
typedef constant const double2* BigTab;
#else
typedef global const double2* BigTab;
#endif

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void
   
void read(u32 WG, u32 N, T2 *u, const global T2 *in, u32 base) {
  in += base + (u32) get_local_id(0);
  for (u32 i = 0; i < N; ++i) { u[i] = in[i * WG]; }
}

void write(u32 WG, u32 N, T2 *u, global T2 *out, u32 base) {
  out += base + (u32) get_local_id(0);
  for (u32 i = 0; i < N; ++i) { out[i * WG] = u[i]; }
}

T2 U2(T a, T b) { return (T2) (a, b); }

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

void OVERLOAD bar(void) {
  // barrier(CLK_LOCAL_MEM_FENCE) is correct, but it turns out that on some GPUs
  // (in particular on Radeon VII and Radeon PRO VII) barrier(0) works as well and is faster.
  // So allow selecting the faster path when it works with -use FAST_BARRIER
#if FAST_BARRIER
  barrier(0);
#else
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
}

void OVERLOAD bar(u32 WG) { if (WG > WAVEFRONT) { bar(); } }

// A half-barrier is only needed when half-a-workgroup needs a barrier.
// This is used e.g. by the double-wide tailSquare, where LDS is split between the halves.
void halfBar() { if (get_enqueued_local_size(0) / 2 > WAVEFRONT) { bar(); } }

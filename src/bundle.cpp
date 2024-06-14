// Copyright (C) Mihai Preda
// Generated file, do not edit. See genbundle.sh and src/cl/*.cl

#include <vector>

static const std::vector<const char*> CL_FILES{
// src/cl/base.cl
R"cltag(
// Copyright (C) Mihai Preda and George Woltman

#pragma once

/* List of user-serviceable -use flags and their effects : see also help (-h)

OUT_WG,OUT_SIZEX <AMD default is 256,32> <nVidia default is 256,4 but needs testing>
IN_WG,IN_SIZEX <AMD default is 256,32>  <nVidia default is 256,4 but needs testing>

UNROLL_WIDTH <nVidia default>
NO_UNROLL_WIDTH <AMD default>
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

// The ROCm optimizer does a very, very poor job of keeping register usage to a minimum.  This negatively impacts occupancy
// which can make a big performance difference.  To counteract this, we can prevent some loops from being unrolled.
// For AMD GPUs we do not unroll fft_WIDTH loops. For nVidia GPUs, we unroll everything.
#if !UNROLL_WIDTH && !NO_UNROLL_WIDTH && !AMDGPU
#define UNROLL_WIDTH 1
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

#if UNROLL_WIDTH
#define UNROLL_WIDTH_CONTROL
#else
#define UNROLL_WIDTH_CONTROL       __attribute__((opencl_unroll_hint(1)))
#endif

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
)cltag",

// src/cl/carry.cl
R"cltag(
// Copyright (C) Mihai Preda

#include "carryutil.cl"
#include "weight.cl"

// Carry propagation with optional MUL-3, over CARRY_LEN words.
// Input arrives conjugated and inverse-weighted.

KERNEL(G_W) carry(P(Word2) out, CP(T2) in, u32 posROE, P(CarryABM) carryOut, CP(u32) bits,
                  BigTab THREAD_WEIGHTS,  P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;

  // & vs. && to workaround spurious warning
  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  // Split 32 bits into CARRY_LEN groups of 2 bits.
#define GPW (16 / CARRY_LEN)
  u32 b = bits[(G_W * g + me) / GPW] >> (me % GPW * (2 * CARRY_LEN));
#undef GPW

  T base = optionalDouble(fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx)));

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;
    double w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + gy * CARRY_LEN + i].x));
    double w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP));
    T2 x = conjugate(in[p]) * U2(w1, w2);
        
#if MUL3
    out[p] = carryPairMul(x, &carry, test(b, 2 * i), test(b, 2 * i + 1), carry, &roundMax, &carryMax);
#else
    out[p] = carryPair(x, &carry, test(b, 2 * i), test(b, 2 * i + 1), carry, &roundMax, &carryMax);
#endif
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}
)cltag",

// src/cl/carryb.cl
R"cltag(
// Copyright (C) Mihai Preda

#include "carryutil.cl"

KERNEL(G_W) carryB(P(Word2) io, CP(CarryABM) carryIn, CP(u32) bits) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);  
  u32 gx = g % NW;
  u32 gy = g / NW;

  // Split 32 bits into CARRY_LEN groups of 2 bits.
#define GPW (16 / CARRY_LEN)
  u32 b = bits[(G_W * g + me) / GPW] >> (me % GPW * (2 * CARRY_LEN));
#undef GPW

  u32 step = G_W * gx + WIDTH * CARRY_LEN * gy;
  io += step;

  u32 HB = BIG_HEIGHT / CARRY_LEN;

  u32 prev = (gy + HB * G_W * gx + HB * me + (HB * WIDTH - 1)) % (HB * WIDTH);
  u32 prevLine = prev % HB;
  u32 prevCol  = prev / HB;

  CarryABM carry = carryIn[WIDTH * prevLine + prevCol];

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = i * WIDTH + me;
    io[p] = carryWord(io[p], &carry, test(b, 2 * i), test(b, 2 * i + 1));
    if (!carry) { return; }
  }
}
)cltag",

// src/cl/carryfused.cl
R"cltag(
// Copyright (C) Mihai Preda

#include "carryutil.cl"
#include "weight.cl"
#include "fftwidth.cl"
#include "middle.cl"

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W) carryFused(P(T2) out, CP(T2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
                       CP(u32) bits, BigTab THREAD_WEIGHTS, P(uint) bufROE) {
  local T2 lds[WIDTH / 2];
  
  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
  u32 line = gr % H;

  T2 u[NW];
  
  readCarryFusedLine(in, u, line);

  // Split 32 bits into NW groups of 2 bits.
#define GPW (16 / NW)
  u32 b = bits[(G_W * line + me) / GPW] >> (me % GPW * (2 * NW));
#undef GPW
  
  fft_WIDTH(lds, u, smallTrig);

  Word2 wu[NW];
  T2 weights = fancyMul(THREAD_WEIGHTS[me], THREAD_WEIGHTS[G_W + line]);

  /*
#if MUL3
  typedef CFMcarry Tcarry;
#else
  typedef CFcarry Tcarry;
#endif

  P(Tcarry) carryShuttlePtr = (P(Tcarry)) carryShuttle;
  Tcarry carry[NW];
*/

#if MUL3
  P(CFMcarry) carryShuttlePtr = (P(CFMcarry)) carryShuttle;
  CFMcarry carry[NW+1];
#else
  P(CFcarry) carryShuttlePtr = (P(CFcarry)) carryShuttle;
  CFcarry carry[NW+1];
#endif

  float roundMax = 0;
  float carryMax = 0;
  
  // Apply the inverse weights

  T invBase = optionalDouble(weights.x);
  
  for (u32 i = 0; i < NW; ++i) {
    T invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)));
    T invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP));

    u[i] = conjugate(u[i]) * U2(invWeight1, invWeight2);
  }

  // Generate our output carries
  for (i32 i = 0; i < NW; ++i) {
#if MUL3
    wu[i] = carryPairMul(u[i], &carry[i], test(b, 2 * i), test(b, 2 * i + 1), 0, &roundMax, &carryMax);    
#else
    wu[i] = carryPair(u[i], &carry[i], test(b, 2 * i), test(b, 2 * i + 1),
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, &roundMax, &carryMax);
#endif
  }

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries. Only groups 0 to H-1 need to write carries out.
  // Group H is a duplicate of group 0 (producing the same results) so we don't care about group H writing out,
  // but it's fine either way.
  if (gr < H) { for (i32 i = 0; i < NW; ++i) { carryShuttlePtr[gr * WIDTH + me * NW + i] = carry[i]; } }

#if OLD_FENCE

  if (gr < H) {
    work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    if (me == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
  }
  if (gr == 0) { return; }
  if (me == 0) { while(!atomic_load((atomic_uint *) &ready[gr - 1])); }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);

#else

  if (gr < H) {
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (me % WAVEFRONT == 0) { atomic_store((atomic_uint *) &ready[gr * (G_W / WAVEFRONT) + me / WAVEFRONT], 1); }
  }
  if (gr == 0) { return; }
  if (me % WAVEFRONT == 0) {
    while(!atomic_load((atomic_uint *) &ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT]));
  }
  read_mem_fence(CLK_GLOBAL_MEM_FENCE);

#endif

  // Read from the carryShuttle carries produced by the previous WIDTH row.  Rotate carries from the last WIDTH row.
  // The new carry layout lets the compiler generate global_load_dwordx4 instructions.
  if (gr < H) {
    for (i32 i = 0; i < NW; ++i) {
      carry[i] = carryShuttlePtr[(gr - 1) * WIDTH + me * NW + i];
    }
  } else {

#if !OLD_FENCE
    // For gr==H we need the barrier since the carry reading is shifted, thus the per-wavefront trick does not apply.
    work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
#endif

    for (i32 i = 0; i < NW; ++i) {
      carry[i] = carryShuttlePtr[(gr - 1) * WIDTH + (me + G_W - 1) % G_W * NW + i /* ((me!=0) + NW - 1 + i) % NW*/];
    }

    if (me == 0) {
      // Tcarry tmp = carry[NW - 1];

      carry[NW] = carry[NW-1];
      for (i32 i = NW-1; i; --i) { carry[i] = carry[i-1]; }
      carry[0] = carry[NW];
    }
  }

  // Apply each 32 or 64 bit carry to the 2 words
  for (i32 i = 0; i < NW; ++i) {
    wu[i] = carryFinal(wu[i], carry[i], test(b, 2 * i));
  }
  
  T base = optionalHalve(weights.y);
  
  for (u32 i = 0; i < NW; ++i) {
    T weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    T weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP));
    u[i] = U2(wu[i].x, wu[i].y) * U2(weight1, weight2);
  }

  bar();

  fft_WIDTH(lds, u, smallTrig);
  write(G_W, NW, u, out, WIDTH * line);

  // Clear carry ready flag for next iteration
#if OLD_FENCE
  if (me == 0) ready[gr - 1] = 0;
#else
  if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
}
)cltag",

// src/cl/carryinc.cl
R"cltag(
// Copyright (C) Mihai Preda

// This file is included with different definitions for iCarry

Word2 OVERLOAD carryPair(T2 u, iCARRY *outCarry, bool b1, bool b2, iCARRY inCarry, float *maxROE, float* carryMax) {
  iCARRY midCarry;
  Word a = carryStep(doubleToLong(u.x, maxROE) + inCarry, &midCarry, b1);
  Word b = carryStep(doubleToLong(u.y, maxROE) + midCarry, outCarry, b2);
// #if STATS & 0x5
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
// #endif
  return (Word2) (a, b);
}

Word2 OVERLOAD carryFinal(Word2 u, iCARRY inCarry, bool b1) {
  i32 tmpCarry;
  u.x = carryStep(u.x + inCarry, &tmpCarry, b1);
  u.y += tmpCarry;
  return u;
}
)cltag",

// src/cl/carryutil.cl
R"cltag(
// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"

#if STATS || ROE
void updateStats(global uint *bufROE, u32 posROE, float roundMax) {
  assert(roundMax >= 0);
  u32 groupRound = work_group_reduce_max(as_uint(roundMax));

  if (get_local_id(0) == 0) { atomic_max(bufROE + posROE, groupRound); }
}
#endif

#if 0 && HAS_ASM
i32  lowBits(i32 u, u32 bits) { i32 tmp; __asm("v_bfe_i32 %0, %1, 0, %2" : "=v" (tmp) : "v" (u), "v" (bits)); return tmp; }
i32 xtract32(i64 x, u32 bits) { i32 tmp; __asm("v_alignbit_b32 %0, %1, %2, %3" : "=v"(tmp) : "v"(as_int2(x).y), "v"(as_int2(x).x), "v"(bits)); return tmp; }
#else
i32  lowBits(i32 u, u32 bits) { return ((u << (32 - bits)) >> (32 - bits)); }
i32 xtract32(i64 x, u32 bits) { return x >> bits; }
#endif

#if !defined(LL)
#define LL 0
#endif

u32 bitlen(bool b) { return EXP / NWORDS + b; }
bool test(u32 bits, u32 pos) { return (bits >> pos) & 1; }

// We support two sizes of carry in carryFused.  A 32-bit carry halves the amount of memory used by CarryShuttle,
// but has some risks.  As FFT sizes increase and/or exponents approach the limit of an FFT size, there is a chance
// that the carry will not fit in 32-bits -- corrupting results.  That said, I did test 2000 iterations of an exponent
// just over 1 billion.  Max(abs(carry)) was 0x637225E9 which is OK (0x80000000 or more is fatal).  P-1 testing is more
// problematic as the mul-by-3 triples the carry too.

// Check for round off errors above a threshold (default is 0.43)
void ROUNDOFF_CHECK(double x) {
#if DEBUG
#ifndef ROUNDOFF_LIMIT
#define ROUNDOFF_LIMIT 0.43
#endif
  float error = fabs(x - rint(x));
  if (error > ROUNDOFF_LIMIT) printf("Roundoff: %g %30.2f\n", error, x);
#endif
}

// Rounding constant: 3 * 2^51, See https://stackoverflow.com/questions/17035464
#define RNDVAL (3.0 * (1l << 51))

i64 doubleToLong(double x, float* maxROE) {
  // Unfortunatelly (i64) rint() is slow!
  // return rint(x);

  // ROUNDOFF_CHECK(x);

  double d = x + RNDVAL;
  float roundoff = fabs((float) (x - (d - RNDVAL)));
  *maxROE = max(*maxROE, roundoff);

  // i32 roundoff = abs(as_int2(x - (d - RNDVAL2)).x);

  int2 words = as_int2(d);

#if EXP / NWORDS >= 19
  // We extend the range to 52 bits instead of 51 by taking the sign from the negation of bit 51
  words.y ^= 0x00080000u;
  words.y = lowBits(words.y, 20);

#if 0
  words.y <<= 12;
  words.y ^= 0x80000000u;
  words.y >>= 12;
#endif
#else
  // Take the sign from bit 50 (i.e. use lower 51 bits).
  words.y = lowBits(words.y, 19);
#endif

  return as_long(words);
}

Word OVERLOAD carryStep(i64 x, i64 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);
  x -= w;
  *outCarry = x >> nBits;
  return w;
}

// Check for 32-bit carry nearing the limit (default is 0x7C000000)
void CARRY32_CHECK(i32 x) {
#if DEBUG
#ifndef CARRY32_LIMIT
#define CARRY32_LIMIT 0x7C000000
#endif
  if (abs(x) > CARRY32_LIMIT) { printf("Carry32: %X\n", abs(x)); }
#endif
}

Word OVERLOAD carryStep(i64 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);
  *outCarry = xtract32(x, nBits) + (w < 0);
  CARRY32_CHECK(*outCarry);
  return w;
}

Word OVERLOAD carryStep(i32 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);
  *outCarry = (x - w) >> nBits;
  CARRY32_CHECK(*outCarry);
  return w;
}

// map abs(carry) to floats, with 2^32 corresponding to 1.0
float OVERLOAD boundCarry(i32 c) { return ldexp(fabs((float) c), -32); }
float OVERLOAD boundCarry(i64 c) { return ldexp(fabs((float) (i32) (c >> 8)), -24); }

#define iCARRY i32
#include "carryinc.cl"
#undef iCARRY

#define iCARRY i64
#include "carryinc.cl"
#undef iCARRY

// In the carryMul situation there's an additional multiplication by 3, adding about 1.6bits to carry, so 32bits
// is often not enough.
typedef i64 CFMcarry;

#if CARRY32
typedef i32 CFcarry;
#else
typedef i64 CFcarry;
#endif

Word2 OVERLOAD carryPairMul(T2 u, i64 *outCarry, bool b1, bool b2, i64 inCarry, float* maxROE, float* carryMax) {
  i64 midCarry;
  Word a = carryStep(3 * doubleToLong(u.x, maxROE) + inCarry, &midCarry, b1);
  Word b = carryStep(3 * doubleToLong(u.y, maxROE) + midCarry, outCarry, b2);
// #if STATS & 0xA
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
// #endif
  return (Word2) (a, b);
}

// Carry propagation from word and carry.
Word2 carryWord(Word2 a, CarryABM* carry, bool b1, bool b2) {
  a.x = carryStep(a.x + *carry, carry, b1);
  a.y = carryStep(a.y + *carry, carry, b2);
  return a;
}
)cltag",

// src/cl/etc.cl
R"cltag(
// Copyright (C) Mihai Preda

#include "base.cl"

#if READRESIDUE
// Read 64 Word2 starting at position 'startDword'.
KERNEL(64) readResidue(P(Word2) out, CP(Word2) in, u32 startDword) {
  u32 me = get_local_id(0);
  u32 k = (startDword + me) % ND;
  u32 y = k % BIG_HEIGHT;
  u32 x = k / BIG_HEIGHT;
  out[me] = in[WIDTH * y + x];
}
#endif

#if SUM64
KERNEL(256) sum64(global ulong* out, u32 sizeBytes, global ulong* in) {
  if (get_global_id(0) == 0) { out[0] = 0; }
  
  ulong sum = 0;
  for (i32 p = get_global_id(0); p < sizeBytes / sizeof(u64); p += get_global_size(0)) {
    sum += in[p];
  }
  sum = work_group_reduce_add(sum);
  if (get_local_id(0) == 0) {
    u32 low = sum;
    u32 prev = atomic_add((global u32*)out, low);
    u32 high = (sum + prev) >> 32;
    atomic_add(((global u32*)out) + 1, high);
  }
}
#endif

#if ISEQUAL
// outEqual must be "true" on entry.
KERNEL(256) isEqual(global i64 *in1, global i64 *in2, P(int) outEqual) {
  for (i32 p = get_global_id(0); p < ND; p += get_global_size(0)) {
    if (in1[p] != in2[p]) {
      *outEqual = 0;
      return;
    }
  }
}
#endif

#if TEST_KERNEL
// Generate a small unused kernel so developers can look at how well individual macros assemble and optimize
kernel void testKernel(global double* in, global float* out) {
  uint me = get_local_id(0);

  double x = in[me];
  double d = x + RNDVAL;
  out[me] = fabs((float) (x + (RNDVAL - d)));
}

#endif
)cltag",

// src/cl/fft-middle.cl
R"cltag(
// Copyright (C) Mihai Preda

#include "trig.cl"

void fft2(T2* u) { X2(u[0], u[1]); }

#if MIDDLE == 3
#include "fft3.cl"
#elif MIDDLE == 4
#include "fft4.cl"
#elif MIDDLE == 5
#include "fft5.cl"
#elif MIDDLE == 6
#include "fft6.cl"
#elif MIDDLE == 7
#include "fft7.cl"
#elif MIDDLE == 8
#include "fft8.cl"
#elif MIDDLE == 9
#include "fft9.cl"
#elif MIDDLE == 10
#include "fft10.cl"
#elif MIDDLE == 11
#include "fft11.cl"
#elif MIDDLE == 12
#include "fft12.cl"
#elif MIDDLE == 13
#include "fft13.cl"
#elif MIDDLE == 14
#include "fft14.cl"
#elif MIDDLE == 15
#include "fft15.cl"
#endif

void fft_MIDDLE(T2 *u) {
#if MIDDLE == 1
  // Do nothing
#elif MIDDLE == 2
  fft2(u);
#elif MIDDLE == 3
  fft3(u);
#elif MIDDLE == 4
  fft4(u);
#elif MIDDLE == 5
  fft5(u);
#elif MIDDLE == 6
  fft6(u);
#elif MIDDLE == 7
  fft7(u);
#elif MIDDLE == 8
  fft8(u);
#elif MIDDLE == 9
  fft9(u);
#elif MIDDLE == 10
  fft10(u);
#elif MIDDLE == 11
  fft11(u);
#elif MIDDLE == 12
  fft12(u);
#elif MIDDLE == 13
  fft13(u);
#elif MIDDLE == 14
  fft14(u);
#elif MIDDLE == 15
  fft15(u);
#else
#error UNRECOGNIZED MIDDLE
#endif
}

// Apply the twiddles needed after fft_MIDDLE and before fft_HEIGHT in forward FFT.
// Also used after fft_HEIGHT and before fft_MIDDLE in inverse FFT.

#define WADD(i, w) u[i] = mul(u[i], w)
#define WSUB(i, w) u[i] = mul_by_conjugate(u[i], w)

#define WADDF(i, w) u[i] = fancyMulTrig(u[i], w)
#define WSUBF(i, w) u[i] = fancyMulTrig(u[i], conjugate(w))

// Keep in sync with TrigBufCache.cpp, see comment there.
#define SHARP_MIDDLE 5

#if !defined(MM_CHAIN) && !defined(MM2_CHAIN) && TRIG_HI
#define MM_CHAIN 1
#define MM2_CHAIN 2
#endif

void middleMul(T2 *u, u32 s, Trig trig, BigTab TRIG_BHW) {
  assert(s < SMALL_HEIGHT);
  if (MIDDLE == 1) { return; }

  T2 w = trig[s];

  if (MIDDLE < SHARP_MIDDLE) {
    WADD(1, w);
#if MM_CHAIN == 0
    T2 base = sq(w);
    for (u32 k = 2; k < MIDDLE; ++k) {
      WADD(k, base);
      base = mul(base, w);
    }
#elif MM_CHAIN == 1
    for (u32 k = 2; k < MIDDLE; ++k) { WADD(k, slowTrig_N(WIDTH * k * s, WIDTH * k * SMALL_HEIGHT, TRIG_BHW)); }
#else
#error MM_CHAIN must be 0 or 1
#endif

  } else { // MIDDLE >= 5

#if MM_CHAIN == 0
    WADDF(1, w);
    T2 base;
    if (MIDDLE >= 10) {
      base = fancySqUpdate(w);
      WADDF(2, base);
      base.x += 1;
    } else {
      base = w;
      base.x += 1;
      base = fancyMulTrig(base, w);
      WADD(2, base);
    }

    for (u32 k = 3; k < MIDDLE; ++k) {
      base = fancyMulTrig(base, w);
      WADD(k, base);
    }

#elif MM_CHAIN == 1
    for (u32 k = 3 + (MIDDLE - 2) % 3; k < MIDDLE; k += 3) {
      T2 base = slowTrig_N(WIDTH * k * s, WIDTH * SMALL_HEIGHT * k, TRIG_BHW);
      WADD(k-1, base);
      WADD(k,   base);
      WADD(k+1, base);
    }

    for (u32 k = 3 + (MIDDLE - 2) % 3; k < MIDDLE; k += 3) {
      WSUBF(k-1, w);
      WADDF(k+1, w);
    }

    WADDF(1, w);

    if ((MIDDLE - 2) % 3 > 0) {
      WADDF(2, w);
      WADDF(2, w);
    }

    if ((MIDDLE - 2) % 3 == 2) {
      WADDF(3, w);
      WADDF(3, fancySqUpdate(w));
    }
#else
#error MM_CHAIN must be 0 or 1.
#endif
  }
}

void middleMul2(T2 *u, u32 x, u32 y, double factor, Trig trig, BigTab TRIG_BHW) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);

  if (MIDDLE == 1) {
    WADD(0, slowTrig_N(x * y, ND / MIDDLE, TRIG_BHW) * factor);
    return;
  }

  T2 w = trig[SMALL_HEIGHT + x];

  if (MIDDLE < SHARP_MIDDLE) {
    T2 base = slowTrig_N(x * y + x * SMALL_HEIGHT, ND / MIDDLE * 2, TRIG_BHW) * factor;
    for (u32 k = 0; k < MIDDLE; ++k) { WADD(k, base); }
    WSUB(0, w);
    if (MIDDLE > 2) { WADD(2, w); }
    if (MIDDLE > 3) { WADD(3, w); WADD(3, w); }

  } else { // MIDDLE >= 5
    // T2 w = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE, TRIG_BHW);

#if MM2_CHAIN == 0

    T2 base = slowTrig_N(x * y + x * SMALL_HEIGHT, ND / MIDDLE * 2, TRIG_BHW) * factor;
    WADD(0, base);
    WADD(1, base);

    for (u32 k = 2; k < MIDDLE; ++k) {
      base = fancyMulTrig(base, w);
      WADD(k, base);
    }
    WSUBF(0, w);

#elif MM2_CHAIN == 1
    u32 cnt = 1;
    for (u32 start = 0, sz = (MIDDLE - start + cnt - 1) / cnt; cnt > 0; --cnt, start += sz) {
      if (start + sz > MIDDLE) { --sz; }
      u32 n = (sz - 1) / 2;
      u32 mid = start + n;

      T2 base1 = slowTrig_N(x * y + x * SMALL_HEIGHT * mid, ND / MIDDLE * (mid + 1), TRIG_BHW) * factor;
      WADD(mid, base1);

      T2 base2 = base1;
      for (u32 i = 1; i <= n; ++i) {
        base1 = fancyMulTrig(base1, conjugate(w));
        WADD(mid - i, base1);

        base2 = fancyMulTrig(base2, w);
        WADD(mid + i, base2);
      }
      if (!(sz & 1)) {
        base2 = fancyMulTrig(base2, w);
        WADD(mid + n + 1, base2);
      }
    }

#elif MM2_CHAIN == 2
    T2 base;
    for (u32 i = 1; i < MIDDLE; i += 3) {
      base = slowTrig_N(x * y + x * SMALL_HEIGHT * i, ND / MIDDLE * (i + 1), TRIG_BHW) * factor;
      WADD(i-1, base);
      WADD(i,   base);
      if (i + 1 < MIDDLE) { WADD(i+1, base); }
    }
    if (MIDDLE % 3 == 1) { WADD(MIDDLE-1, base); }
    for (u32 i = 0; i + 1 < MIDDLE; i += 3) { WSUBF(i, w); }
    for (u32 i = 2; i < MIDDLE; i += 3) { WADDF(i, w); }
    if (MIDDLE % 3 == 1) { WADDF(MIDDLE-1, w); WADDF(MIDDLE-1, w); }
#else
#error MM2_CHAIN must be 0, 1 or 2.
#endif
  }
}

#undef WADD
#undef WADDF
#undef WSUB
#undef WSUBF

// Do a partial transpose during fftMiddleIn/Out
// The AMD OpenCL optimization guide indicates that reading/writing T values will be more efficient
// than reading/writing T2 values.  This routine lets us try both versions.

void middleShuffle(local T *lds, T2 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  if (MIDDLE <= 8) {
    local T *p = lds + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    for (int i = 0; i < MIDDLE; ++i) { p[i * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].x = lds[me + workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p[i * workgroupSize] = u[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].y = lds[me + workgroupSize * i]; }
  } else {
    local int *p1 = ((local int*) lds) + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    local int *p2 = (local int*) lds;
    int4 *pu = (int4 *)u;

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].x = p2[me + workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].y = p2[me + workgroupSize * i]; }
    bar();

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].z; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].z = p2[me + workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].w; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].w = p2[me + workgroupSize * i]; }
  }
}
)cltag",

// src/cl/fft10.cl
R"cltag(
// See prime95's gwnum/zr10.mac file for more detailed explanation of the formulas below
//R1 = (r1+r6)     +((r2+r7)+(r5+r10))     +((r3+r8)+(r4+r9))
//R3 = (r1+r6) +.309((r2+r7)+(r5+r10)) -.809((r3+r8)+(r4+r9)) +.951((i2+i7)-(i5+i10)) +.588((i3+i8)-(i4+i9))
//R9 = (r1+r6) +.309((r2+r7)+(r5+r10)) -.809((r3+r8)+(r4+r9)) -.951((i2+i7)-(i5+i10)) -.588((i3+i8)-(i4+i9))
//R5 = (r1+r6) -.809((r2+r7)+(r5+r10)) +.309((r3+r8)+(r4+r9)) +.588((i2+i7)-(i5+i10)) -.951((i3+i8)-(i4+i9))
//R7 = (r1+r6) -.809((r2+r7)+(r5+r10)) +.309((r3+r8)+(r4+r9)) -.588((i2+i7)-(i5+i10)) +.951((i3+i8)-(i4+i9))
//R6 = (r1-r6)     -((r2-r7)-(r5-r10))     +((r3-r8)-(r4-r9))
//R2 = (r1-r6) +.809((r2-r7)-(r5-r10)) +.309((r3-r8)-(r4-r9)) +.588((i2-i7)+(i5-i10)) +.951((i3-i8)+(i4-i9))
//R10= (r1-r6) +.809((r2-r7)-(r5-r10)) +.309((r3-r8)-(r4-r9)) -.588((i2-i7)+(i5-i10)) -.951((i3-i8)+(i4-i9))
//R4 = (r1-r6) -.309((r2-r7)-(r5-r10)) -.809((r3-r8)-(r4-r9)) +.951((i2-i7)+(i5-i10)) -.588((i3-i8)+(i4-i9))
//R8 = (r1-r6) -.309((r2-r7)-(r5-r10)) -.809((r3-r8)-(r4-r9)) -.951((i2-i7)+(i5-i10)) +.588((i3-i8)+(i4-i9))

//I1 = (i1+i6)     +((i2+i7)+(i5+i10))     +((i3+i8)+(i4+i9))
//I3 = (i1+i6) +.309((i2+i7)+(i5+i10)) -.809((i3+i8)+(i4+i9)) -.951((r2+r7)-(r5+r10)) -.588((r3+r8)-(r4+r9))
//I9 = (i1+i6) +.309((i2+i7)+(i5+i10)) -.809((i3+i8)+(i4+i9)) +.951((r2+r7)-(r5+r10)) +.588((r3+r8)-(r4+r9))
//I5 = (i1+i6) -.809((i2+i7)+(i5+i10)) +.309((i3+i8)+(i4+i9)) -.588((r2+r7)-(r5+r10)) +.951((r3+r8)-(r4+r9))
//I7 = (i1+i6) -.809((i2+i7)+(i5+i10)) +.309((i3+i8)+(i4+i9)) +.588((r2+r7)-(r5+r10)) -.951((r3+r8)-(r4+r9))
//I6 = (i1-i6)     -((i2-i7)-(i5-i10))     +((i3-i8)-(i4-i9))
//I2 = (i1-i6) +.809((i2-i7)-(i5-i10)) +.309((i3-i8)-(i4-i9)) -.588((r2-r7)+(r5-r10)) -.951((r3-r8)+(r4-r9))
//I10= (i1-i6) +.809((i2-i7)-(i5-i10)) +.309((i3-i8)-(i4-i9)) +.588((r2-r7)+(r5-r10)) +.951((r3-r8)+(r4-r9))
//I4 = (i1-i6) -.309((i2-i7)-(i5-i10)) -.809((i3-i8)-(i4-i9)) -.951((r2-r7)+(r5-r10)) +.588((r3-r8)+(r4-r9))
//I8 = (i1-i6) -.309((i2-i7)-(i5-i10)) -.809((i3-i8)-(i4-i9)) +.951((r2-r7)+(r5-r10)) -.588((r3-r8)+(r4-r9))

void fft10(T2 *u) {
  const double SIN1 = 0x1.e6f0e134454ffp-1;		// sin(tau/5), 0.95105651629515353118
  const double SIN2_SIN1 = 0.618033988749894848;	// sin(2*tau/5) / sin(tau/5) = .588/.951, 0.618033988749894848
  const double COS1 = 0.309016994374947424;		// cos(tau/5), 0.309016994374947424
  const double COS2 = -0.809016994374947424;		// cos(2*tau/5), 0.809016994374947424

  X2(u[0], u[5]);					// (r1+ i1+),  (r1-  i1-)
  X2_mul_t4(u[1], u[6]);				// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[7]);				// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(u[3], u[8]);				// (r4+ i4+),  (i4- -r4-)
  X2_mul_t4(u[4], u[9]);				// (r5+ i5+),  (i5- -r5-)

  X2_mul_t4(u[1], u[4]);				// (r2++  i2++),  (i2+- -r2+-)
  X2_mul_t4(u[2], u[3]);				// (r3++  i3++),  (i3+- -r3+-)
  X2_mul_t4(u[6], u[9]);				// (i2-+ -r2-+), (-r2-- -i2--)
  X2_mul_t4(u[7], u[8]);				// (i3-+ -r3-+), (-r3-- -i3--)

  T2 tmp39a = fmaT2(COS1, u[1], u[0]);
  T2 tmp57a = fmaT2(COS2, u[1], u[0]);
  u[0] = u[0] + u[1];
  T2 tmp210a = fmaT2(COS2, u[9], u[5]);
  T2 tmp48a = fmaT2(COS1, u[9], u[5]);
  u[5] = u[5] + u[9];

  tmp39a = fmaT2(COS2, u[2], tmp39a);
  tmp57a = fmaT2(COS1, u[2], tmp57a);
  u[0] = u[0] + u[2];
  tmp210a = fmaT2(COS1, -u[8], tmp210a);
  tmp48a = fmaT2(COS2, -u[8], tmp48a);
  u[5] = u[5] - u[8];

  T2 tmp39b = fmaT2(SIN2_SIN1, u[3], u[4]);		// (i2+- +.588/.951*i3+-, -r2+- -.588/.951*r3+-)
  T2 tmp57b = fmaT2(SIN2_SIN1, u[4], -u[3]);		// (.588/.951*i2+- -i3+-, -.588/.951*r2+- +r3+-)
  T2 tmp210b = fmaT2(SIN2_SIN1, u[6], u[7]);		// (.588/.951*i2-+ +i3-+, -.588/.951*r2-+ -r3-+)
  T2 tmp48b = fmaT2(SIN2_SIN1, -u[7], u[6]);		// (i2-+ -.588/.951*i3-+, -r2-+ +.588/.951*r3-+)

  fma_addsub(u[1], u[9], SIN1, tmp210a, tmp210b);
  fma_addsub(u[2], u[8], SIN1, tmp39a, tmp39b);
  fma_addsub(u[3], u[7], SIN1, tmp48a, tmp48b);
  fma_addsub(u[4], u[6], SIN1, tmp57a, tmp57b);
}
)cltag",

// src/cl/fft11.cl
R"cltag(
// See prime95's gwnum/zr11.mac file for more detailed explanation of the formulas below
// R1 = r1     +(r2+r11)     +(r3+r10)     +(r4+r9)     +(r5+r8)     +(r6+r7)
// R2 = r1 +.841(r2+r11) +.415(r3+r10) -.142(r4+r9) -.655(r5+r8) -.959(r6+r7)  +(.541(i2-i11) +.910(i3-i10) +.990(i4-i9) +.756(i5-i8) +.282(i6-i7))
// R11= r1 +.841(r2+r11) +.415(r3+r10) -.142(r4+r9) -.655(r5+r8) -.959(r6+r7)  -(.541(i2-i11) +.910(i3-i10) +.990(i4-i9) +.756(i5-i8) +.282(i6-i7))
// R3 = r1 +.415(r2+r11) -.655(r3+r10) -.959(r4+r9) -.142(r5+r8) +.841(r6+r7)  +(.910(i2-i11) +.756(i3-i10) -.282(i4-i9) -.990(i5-i8) -.541(i6-i7))
// R10= r1 +.415(r2+r11) -.655(r3+r10) -.959(r4+r9) -.142(r5+r8) +.841(r6+r7)  -(.910(i2-i11) +.756(i3-i10) -.282(i4-i9) -.990(i5-i8) -.541(i6-i7))
// R4 = r1 -.142(r2+r11) -.959(r3+r10) +.415(r4+r9) +.841(r5+r8) -.655(r6+r7)  +(.990(i2-i11) -.282(i3-i10) -.910(i4-i9) +.541(i5-i8) +.756(i6-i7))
// R9 = r1 -.142(r2+r11) -.959(r3+r10) +.415(r4+r9) +.841(r5+r8) -.655(r6+r7)  -(.990(i2-i11) -.282(i3-i10) -.910(i4-i9) +.541(i5-i8) +.756(i6-i7))
// R5 = r1 -.655(r2+r11) -.142(r3+r10) +.841(r4+r9) -.959(r5+r8) +.415(r6+r7)  +(.756(i2-i11) -.990(i3-i10) +.541(i4-i9) +.282(i5-i8) -.910(i6-i7))
// R8 = r1 -.655(r2+r11) -.142(r3+r10) +.841(r4+r9) -.959(r5+r8) +.415(r6+r7)  -(.756(i2-i11) -.990(i3-i10) +.541(i4-i9) +.282(i5-i8) -.910(i6-i7))
// R6 = r1 -.959(r2+r11) +.841(r3+r10) -.655(r4+r9) +.415(r5+r8) -.142(r6+r7)  +(.282(i2-i11) -.541(i3-i10) +.756(i4-i9) -.910(i5-i8) +.990(i6-i7))
// R7 = r1 -.959(r2+r11) +.841(r3+r10) -.655(r4+r9) +.415(r5+r8) -.142(r6+r7)  -(.282(i2-i11) -.541(i3-i10) +.756(i4-i9) -.910(i5-i8) +.990(i6-i7))

// I1 = i1     +(i2+i11)     +(i3+i10)     +(i4+i9)     +(i5+i8)     +(i6+i7)
// I2 = i1 +.841(i2+i11) +.415(i3+i10) -.142(i4+i9) -.655(i5+i8) -.959(i6+i7)  -(.541(r2-r11) +.910(r3-r10) +.990(r4-r9) +.756(r5-r8) +.282(r6-r7))
// I11= i1 +.841(i2+i11) +.415(i3+i10) -.142(i4+i9) -.655(i5+i8) -.959(i6+i7)  +(.541(r2-r11) +.910(r3-r10) +.990(r4-r9) +.756(r5-r8) +.282(r6-r7))
// I3 = i1 +.415(i2+i11) -.655(i3+i10) -.959(i4+i9) -.142(i5+i8) +.841(i6+i7)  -(.910(r2-r11) +.756(r3-r10) -.282(r4-r9) -.990(r5-r8) -.541(r6-r7))
// I10= i1 +.415(i2+i11) -.655(i3+i10) -.959(i4+i9) -.142(i5+i8) +.841(i6+i7)  +(.910(r2-r11) +.756(r3-r10) -.282(r4-r9) -.990(r5-r8) -.541(r6-r7))
// I4 = i1 -.142(i2+i11) -.959(i3+i10) +.415(i4+i9) +.841(i5+i8) -.655(i6+i7)  -(.990(r2-r11) -.282(r3-r10) -.910(r4-r9) +.541(r5-r8) +.756(r6-r7))
// I9 = i1 -.142(i2+i11) -.959(i3+i10) +.415(i4+i9) +.841(i5+i8) -.655(i6+i7)  +(.990(r2-r11) -.282(r3-r10) -.910(r4-r9) +.541(r5-r8) +.756(r6-r7))
// I5 = i1 -.655(i2+i11) -.142(i3+i10) +.841(i4+i9) -.959(i5+i8) +.415(i6+i7)  -(.756(r2-r11) -.990(r3-r10) +.541(r4-r9) +.282(r5-r8) -.910(r6-r7))
// I8 = i1 -.655(i2+i11) -.142(i3+i10) +.841(i4+i9) -.959(i5+i8) +.415(i6+i7)  +(.756(r2-r11) -.990(r3-r10) +.541(r4-r9) +.282(r5-r8) -.910(r6-r7))
// I6 = i1 -.959(i2+i11) +.841(i3+i10) -.655(i4+i9) +.415(i5+i8) -.142(i6+i7)  -(.282(r2-r11) -.541(r3-r10) +.756(r4-r9) -.910(r5-r8) +.990(r6-r7))
// I7 = i1 -.959(i2+i11) +.841(i3+i10) -.655(i4+i9) +.415(i5+i8) -.142(i6+i7)  +(.282(r2-r11) -.541(r3-r10) +.756(r4-r9) -.910(r5-r8) +.990(r6-r7))

void fft11(T2 *u) {
  const double COS1 = 0.8412535328311811688;		// cos(tau/11)
  const double COS2 = 0.4154150130018864255;		// cos(2*tau/11)
  const double COS3 = -0.1423148382732851404;		// cos(3*tau/11)
  const double COS4 = -0.6548607339452850640;		// cos(4*tau/11)
  const double COS5 = -0.9594929736144973898;		// cos(5*tau/11)
  const double SIN1 = 0.5406408174555975821;		// sin(tau/11)
  const double SIN2_SIN1 = 1.682507065662362337;	// sin(2*tau/11) / sin(tau/11) = .910/.541
  const double SIN3_SIN1 = 1.830830026003772851;	// sin(3*tau/11) / sin(tau/11) = .990/.541
  const double SIN4_SIN1 = 1.397877389115792056;	// sin(4*tau/11) / sin(tau/11) = .756/.541
  const double SIN5_SIN1 = 0.521108558113202723;	// sin(5*tau/11) / sin(tau/11) = .282/.541

  X2_mul_t4(u[1], u[10]);				// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[9]);				// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(u[3], u[8]);				// (r4+ i4+),  (i4- -r4-)
  X2_mul_t4(u[4], u[7]);				// (r5+ i5+),  (i5- -r5-)
  X2_mul_t4(u[5], u[6]);				// (r6+ i6+),  (i6- -r6-)

  T2 tmp211a = fmaT2(COS1, u[1], u[0]);
  T2 tmp310a = fmaT2(COS2, u[1], u[0]);
  T2 tmp49a = fmaT2(COS3, u[1], u[0]);
  T2 tmp58a = fmaT2(COS4, u[1], u[0]);
  T2 tmp67a = fmaT2(COS5, u[1], u[0]);
  u[0] = u[0] + u[1];

  tmp211a = fmaT2(COS2, u[2], tmp211a);
  tmp310a = fmaT2(COS4, u[2], tmp310a);
  tmp49a = fmaT2(COS5, u[2], tmp49a);
  tmp58a = fmaT2(COS3, u[2], tmp58a);
  tmp67a = fmaT2(COS1, u[2], tmp67a);
  u[0] = u[0] + u[2];

  tmp211a = fmaT2(COS3, u[3], tmp211a);
  tmp310a = fmaT2(COS5, u[3], tmp310a);
  tmp49a = fmaT2(COS2, u[3], tmp49a);
  tmp58a = fmaT2(COS1, u[3], tmp58a);
  tmp67a = fmaT2(COS4, u[3], tmp67a);
  u[0] = u[0] + u[3];

  tmp211a = fmaT2(COS4, u[4], tmp211a);
  tmp310a = fmaT2(COS3, u[4], tmp310a);
  tmp49a = fmaT2(COS1, u[4], tmp49a);
  tmp58a = fmaT2(COS5, u[4], tmp58a);
  tmp67a = fmaT2(COS2, u[4], tmp67a);
  u[0] = u[0] + u[4];

  tmp211a = fmaT2(COS5, u[5], tmp211a);
  tmp310a = fmaT2(COS1, u[5], tmp310a);
  tmp49a = fmaT2(COS4, u[5], tmp49a);
  tmp58a = fmaT2(COS2, u[5], tmp58a);
  tmp67a = fmaT2(COS3, u[5], tmp67a);
  u[0] = u[0] + u[5];

  T2 tmp211b = fmaT2(SIN2_SIN1, u[9], u[10]);		// .910/.541
  T2 tmp310b = fmaT2(SIN2_SIN1, u[10], -u[6]);
  T2 tmp49b = fmaT2(SIN2_SIN1, -u[8], u[7]);
  T2 tmp58b = fmaT2(SIN2_SIN1, -u[6], u[8]);
  T2 tmp67b = fmaT2(SIN2_SIN1, -u[7], -u[9]);

  tmp211b = fmaT2(SIN3_SIN1, u[8], tmp211b);		// .990/.541
  tmp310b = fmaT2(SIN3_SIN1, -u[7], tmp310b);
  tmp49b = fmaT2(SIN3_SIN1, u[10], tmp49b);
  tmp58b = fmaT2(SIN3_SIN1, -u[9], tmp58b);
  tmp67b = fmaT2(SIN3_SIN1, u[6], tmp67b);

  tmp211b = fmaT2(SIN4_SIN1, u[7], tmp211b);		// .756/.541
  tmp310b = fmaT2(SIN4_SIN1, u[9], tmp310b);
  tmp49b = fmaT2(SIN4_SIN1, u[6], tmp49b);
  tmp58b = fmaT2(SIN4_SIN1, u[10], tmp58b);
  tmp67b = fmaT2(SIN4_SIN1, u[8], tmp67b);

  tmp211b = fmaT2(SIN5_SIN1, u[6], tmp211b);		// .282/.541
  tmp310b = fmaT2(SIN5_SIN1, -u[8], tmp310b);
  tmp49b = fmaT2(SIN5_SIN1, -u[9], tmp49b);
  tmp58b = fmaT2(SIN5_SIN1, u[7], tmp58b);
  tmp67b = fmaT2(SIN5_SIN1, u[10], tmp67b);

  fma_addsub(u[1], u[10], SIN1, tmp211a, tmp211b);
  fma_addsub(u[2], u[9], SIN1, tmp310a, tmp310b);
  fma_addsub(u[3], u[8], SIN1, tmp49a, tmp49b);
  fma_addsub(u[4], u[7], SIN1, tmp58a, tmp58b);
  fma_addsub(u[5], u[6], SIN1, tmp67a, tmp67b);
}
)cltag",

// src/cl/fft12.cl
R"cltag(
// See prime95's gwnum/zr12.mac file for more detailed explanation of the formulas below
// R1 = (r1+r7)+(r4+r10)     +(((r3+r9)+(r5+r11))+((r2+r8)+(r6+r12)))
// R7 = (r1+r7)-(r4+r10)     +(((r3+r9)+(r5+r11))-((r2+r8)+(r6+r12)))
// R5 = (r1+r7)+(r4+r10) -.500(((r3+r9)+(r5+r11))+((r2+r8)+(r6+r12))) -.866(((i3+i9)-(i5+i11))-((i2+i8)-(i6+i12)))
// R9 = (r1+r7)+(r4+r10) -.500(((r3+r9)+(r5+r11))+((r2+r8)+(r6+r12))) +.866(((i3+i9)-(i5+i11))-((i2+i8)-(i6+i12)))
// R3 = (r1+r7)-(r4+r10) -.500(((r3+r9)+(r5+r11))-((r2+r8)+(r6+r12))) +.866(((i3+i9)-(i5+i11))+((i2+i8)-(i6+i12)))
// R11= (r1+r7)-(r4+r10) -.500(((r3+r9)+(r5+r11))-((r2+r8)+(r6+r12))) -.866(((i3+i9)-(i5+i11))+((i2+i8)-(i6+i12)))
// I1 = (i1+i7)+(i4+i10)     +(((i3+i9)+(i5+i11))+((i2+i8)+(i6+i12)))
// I7 = (i1+i7)-(i4+i10)     +(((i3+i9)+(i5+i11))-((i2+i8)+(i6+i12)))
// I5 = (i1+i7)+(i4+i10) -.500(((i3+i9)+(i5+i11))+((i2+i8)+(i6+i12))) +.866(((r3+r9)-(r5+r11))-((r2+r8)-(r6+r12)))
// I9 = (i1+i7)+(i4+i10) -.500(((i3+i9)+(i5+i11))+((i2+i8)+(i6+i12))) -.866(((r3+r9)-(r5+r11))-((r2+r8)-(r6+r12)))
// I3 = (i1+i7)-(i4+i10) -.500(((i3+i9)+(i5+i11))-((i2+i8)+(i6+i12))) -.866(((r3+r9)-(r5+r11))+((r2+r8)-(r6+r12)))
// I11= (i1+i7)-(i4+i10) -.500(((i3+i9)+(i5+i11))-((i2+i8)+(i6+i12))) +.866(((r3+r9)-(r5+r11))+((r2+r8)-(r6+r12)))

// R4 = (r1-r7)     -((r3-r9)-(r5-r11))				-(i4-i10)     +((i2-i8)+(i6-i12))
// R10= (r1-r7)     -((r3-r9)-(r5-r11))				+(i4-i10)     -((i2-i8)+(i6-i12))
// R2 = (r1-r7) +.500((r3-r9)-(r5-r11)) +.866((r2-r8)-(r6-r12))	+(i4-i10) +.500((i2-i8)+(i6-i12)) +.866((i3-i9)+(i5-i11))
// R12= (r1-r7) +.500((r3-r9)-(r5-r11)) +.866((r2-r8)-(r6-r12))	-(i4-i10) -.500((i2-i8)+(i6-i12)) -.866((i3-i9)+(i5-i11))
// R6 = (r1-r7) +.500((r3-r9)-(r5-r11)) -.866((r2-r8)-(r6-r12))	+(i4-i10) +.500((i2-i8)+(i6-i12)) -.866((i3-i9)+(i5-i11))
// R8 = (r1-r7) +.500((r3-r9)-(r5-r11)) -.866((r2-r8)-(r6-r12))	-(i4-i10) -.500((i2-i8)+(i6-i12)) +.866((i3-i9)+(i5-i11))
// I4 = (i1-i7)     -((i3-i9)-(i5-i11))                         +(r4-r10)     -((r2-r8)+(r6-r12))
// I10= (i1-i7)     -((i3-i9)-(i5-i11))                         -(r4-r10)     +((r2-r8)+(r6-r12))
// I2 = (i1-i7) +.500((i3-i9)-(i5-i11)) +.866((i2-i8)-(i6-i12))	-(r4-r10) -.500((r2-r8)+(r6-r12)) -.866((r3-r9)+(r5-r11))
// I12= (i1-i7) +.500((i3-i9)-(i5-i11)) +.866((i2-i8)-(i6-i12))	+(r4-r10) +.500((r2-r8)+(r6-r12)) +.866((r3-r9)+(r5-r11))
// I6 = (i1-i7) +.500((i3-i9)-(i5-i11)) -.866((i2-i8)-(i6-i12))	-(r4-r10) -.500((r2-r8)+(r6-r12)) +.866((r3-r9)+(r5-r11))
// I8 = (i1-i7) +.500((i3-i9)-(i5-i11)) -.866((i2-i8)-(i6-i12))	+(r4-r10) +.500((r2-r8)+(r6-r12)) -.866((r3-r9)+(r5-r11))

void fft12(T2 *u) {
  const double SIN1 = 0x1.bb67ae8584caap-1;	// sin(tau/3), 0.86602540378443859659;
  const double COS1 = 0.5;			// cos(tau/3)

  X2(u[0], u[6]);				// (r1+ i1+),  (r1-  i1-)
  X2_mul_t4(u[3], u[9]);			// (r4+ i4+),  (i4- -r4-)
  X2_mul_t4(u[1], u[7]);			// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[5], u[11]);			// (r6+ i6+),  (i6- -r6-)
  X2_mul_t4(u[2], u[8]);			// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(u[4], u[10]);			// (r5+ i5+),  (i5- -r5-)

  X2(u[0], u[3]);				// (r1++  i1++),  (r1+- i1+-)
  X2_mul_t4(u[1], u[5]);			// (r2++  i2++),  (i2+- -r2+-)
  X2_mul_t4(u[2], u[4]);			// (r3++  i3++),  (i3+- -r3+-)

  X2_mul_t4(u[7], u[11]);			// (i2-+ -r2-+), (-r2-- -i2--)
  X2_mul_t4(u[8], u[10]);			// (i3-+ -r3-+), (-r3-- -i3--)

  X2(u[2], u[1]);				// (r3+++  i3+++),  (r3++- i3++-)
  X2(u[4], u[5]);				// (i3+-+  -r3+-+), (i3+-- -r3+--)

  T2 tmp26812b = fmaT2(COS1, u[7], u[9]);
  T2 tmp410b = u[9] - u[7];

  T2 tmp26812a = fmaT2(-COS1, u[10], u[6]);
  T2 tmp410a = u[6] + u[10];

  T2 tmp68a, tmp68b, tmp212a, tmp212b;
  fma_addsub(tmp212b, tmp68b, SIN1, tmp26812b, u[8]);
  fma_addsub(tmp68a, tmp212a, SIN1, tmp26812a, u[11]);

  T2 tmp311 = fmaT2(-COS1, u[1], u[3]);
  u[6] = u[3] + u[1];

  T2 tmp59 = fmaT2(-COS1, u[2], u[0]);
  u[0] = u[0] + u[2];

  u[3] = tmp410a - tmp410b;
  u[9] = tmp410a + tmp410b;
  u[1] = tmp212a + tmp212b;
  u[11] = tmp212a - tmp212b;

  fma_addsub(u[2], u[10], SIN1, tmp311, u[4]);
  fma_addsub(u[8], u[4], SIN1, tmp59, u[5]);

  u[5] = tmp68a + tmp68b;
  u[7] = tmp68a - tmp68b;
}
)cltag",

// src/cl/fft13.cl
R"cltag(
// Copyright (C) Mihai Preda and George Woltman

// To calculate a 13-complex FFT in a brute force way (using a shorthand notation):
// The sin/cos values (w = 13th root of unity) are:
// w^1 = .885 - .465i
// w^2 = .568 - .823i
// w^3 = .121 - .993i
// w^4 = -.355 - .935i
// w^5 = -.749 - .663i
// w^6 = -.971 - .239i
// w^7 = -.971 + .239i
// w^8 = -.749 + .663i
// w^9 = -.355 + .935i
// w^10= .121 + .993i
// w^11= .568 + .823i
// w^12= .885 + .465i
//
// R1 = r1     +(r2+r13)     +(r3+r12)     +(r4+r11)     +(r5+r10)     +(r6+r9)     +(r7+r8)
// R2 = r1 +.885(r2+r13) +.568(r3+r12) +.121(r4+r11) -.355(r5+r10) -.749(r6+r9) -.971(r7+r8)  +.465(i2-i13) +.823(i3-i12) +.993(i4-i11) +.935(i5-i10) +.663(i6-i9) +.239(i7-i8)
// R13= r1 +.885(r2+r13) +.568(r3+r12) +.121(r4+r11) -.355(r5+r10) -.749(r6+r9) -.971(r7+r8)  -.465(i2-i13) -.823(i3-i12) -.993(i4-i11) -.935(i5-i10) -.663(i6-i9) -.239(i7-i8)
// R3 = r1 +.568(r2+r13) -.355(r3+r12) -.971(r4+r11) -.749(r5+r10) +.121(r6+r9) +.885(r7+r8)  +.823(i2-i13) +.935(i3-i12) +.239(i4-i11) -.663(i5-i10) -.993(i6-i9) -.465(i7-i8)
// R12= r1 +.568(r2+r13) -.355(r3+r12) -.971(r4+r11) -.749(r5+r10) +.121(r6+r9) +.885(r7+r8)  -.823(i2-i13) -.935(i3-i12) -.239(i4-i11) +.663(i5-i10) +.993(i6-i9) +.465(i7-i8)
// R4 = r1 +.121(r2+r13) -.971(r3+r12) -.355(r4+r11) +.885(r5+r10) +.568(r6+r9) -.749(r7+r8)  +.993(i2-i13) +.239(i3-i12) -.935(i4-i11) -.465(i5-i10) +.823(i6-i9) +.663(i7-i8)
// R11= r1 +.121(r2+r13) -.971(r3+r12) -.355(r4+r11) +.885(r5+r10) +.568(r6+r9) -.749(r7+r8)  -.993(i2-i13) -.239(i3-i12) +.935(i4-i11) +.465(i5-i10) -.823(i6-i9) -.663(i7-i8)
// R5 = r1 -.355(r2+r13) -.749(r3+r12) +.885(r4+r11) +.121(r5+r10) -.971(r6+r9) +.568(r7+r8)  +.935(i2-i13) -.663(i3-i12) -.465(i4-i11) +.993(i5-i10) -.239(i6-i9) -.823(i7-i8)
// R10= r1 -.355(r2+r13) -.749(r3+r12) +.885(r4+r11) +.121(r5+r10) -.971(r6+r9) +.568(r7+r8)  -.935(i2-i13) +.663(i3-i12) +.465(i4-i11) -.993(i5-i10) +.239(i6-i9) +.823(i7-i8)
// R6 = r1 -.749(r2+r13) +.121(r3+r12) +.568(r4+r11) -.971(r5+r10) +.885(r6+r9) -.355(r7+r8)  +.663(i2-i13) -.993(i3-i12) +.823(i4-i11) -.239(i5-i10) -.465(i6-i9) +.935(i7-i8)
// R9 = r1 -.749(r2+r13) +.121(r3+r12) +.568(r4+r11) -.971(r5+r10) +.885(r6+r9) -.355(r7+r8)  -.663(i2-i13) +.993(i3-i12) -.823(i4-i11) +.239(i5-i10) +.465(i6-i9) -.935(i7-i8)
// R7 = r1 -.971(r2+r13) +.885(r3+r12) -.749(r4+r11) +.568(r5+r10) -.355(r6+r9) +.121(r7+r8)  +.239(i2-i13) -.465(i3-i12) +.663(i4-i11) -.823(i5-i10) +.935(i6-i9) -.993(i7-i8)
// R8 = r1 -.971(r2+r13) +.885(r3+r12) -.749(r4+r11) +.568(r5+r10) -.355(r6+r9) +.121(r7+r8)  -.239(i2-i13) +.465(i3-i12) -.663(i4-i11) +.823(i5-i10) -.935(i6-i9) +.993(i7-i8)
//
// I1 = i1                                                                                        +(i2+i13)     +(i3+i12)     +(i4+i11)     +(i5+i10)     +(i6+i9)     +(i7+i8)
// I2 = i1 -.465(r2-r13) -.823(r3-r12) -.993(r4-r11) -.935(r5-r10) -.663(r6-r9) -.239(r7-r8)  +.885(i2+i13) +.568(i3+i12) +.121(i4+i11) -.355(i5+i10) -.749(i6+i9) -.971(i7+i8)
// I13= i1 +.465(r2-r13) +.823(r3-r12) +.993(r4-r11) +.935(r5-r10) +.663(r6-r9) +.239(r7-r8)  +.885(i2+i13) +.568(i3+i12) +.121(i4+i11) -.355(i5+i10) -.749(i6+i9) -.971(i7+i8)
// I3 = i1 -.823(r2-r13) -.935(r3-r12) -.239(r4-r11) +.663(r5-r10) +.993(r6-r9) +.465(r7-r8)  +.568(i2+i13) -.355(i3+i12) -.971(i4+i11) -.749(i5+i10) +.121(i6+i9) +.885(i7+i8)
// I12= i1 +.823(r2-r13) +.935(r3-r12) +.239(r4-r11) -.663(r5-r10) -.993(r6-r9) -.465(r7-r8)  +.568(i2+i13) -.355(i3+i12) -.971(i4+i11) -.749(i5+i10) +.121(i6+i9) +.885(i7+i8)
// I4 = i1 -.993(r2-r13) -.239(r3-r12) +.935(r4-r11) +.465(r5-r10) -.823(r6-r9) -.663(r7-r8)  +.121(i2+i13) -.971(i3+i12) -.355(i4+i11) +.885(i5+i10) +.568(i6+i9) -.749(i7+i8)
// I11= i1 +.993(r2-r13) +.239(r3-r12) -.935(r4-r11) -.465(r5-r10) +.823(r6-r9) +.663(r7-r8)  +.121(i2+i13) -.971(i3+i12) -.355(i4+i11) +.885(i5+i10) +.568(i6+i9) -.749(i7+i8)
// I5 = i1 -.935(r2-r13) +.663(r3-r12) +.465(r4-r11) -.993(r5-r10) +.239(r6-r9) +.823(r7-r8)  -.355(i2+i13) -.749(i3+i12) +.885(i4+i11) +.121(i5+i10) -.971(i6+i9) +.568(i7+i8)
// I10= i1 +.935(r2-r13) -.663(r3-r12) -.465(r4-r11) +.993(r5-r10) -.239(r6-r9) -.823(r7-r8)  -.355(i2+i13) -.749(i3+i12) +.885(i4+i11) +.121(i5+i10) -.971(i6+i9) +.568(i7+i8)
// I6 = i1 -.663(r2-r13) +.993(r3-r12) -.823(r4-r11) +.239(r5-r10) +.465(r6-r9) -.993(r7-r8)  -.749(i2+i13) +.121(i3+i12) +.568(i4+i11) -.971(i5+i10) +.885(i6+i9) -.355(i7+i8)
// I9 = i1 +.663(r2-r13) -.993(r3-r12) +.823(r4-r11) -.239(r5-r10) -.465(r6-r9) +.993(r7-r8)  -.749(i2+i13) +.121(i3+i12) +.568(i4+i11) -.971(i5+i10) +.885(i6+i9) -.355(i7+i8)
// I7 = i1 -.239(r2-r13) +.465(r3-r12) -.663(r4-r11) +.823(r5-r10) -.935(r6-r9) +.935(r7-r8)  -.971(i2+i13) +.885(i3+i12) -.749(i4+i11) +.568(i5+i10) -.355(i6+i9) +.121(i7+i8)
// I8 = i1 +.239(r2-r13) -.465(r3-r12) +.663(r4-r11) -.823(r5-r10) +.935(r6-r9) -.935(r7-r8)  -.971(i2+i13) +.885(i3+i12) -.749(i4+i11) +.568(i5+i10) -.355(i6+i9) +.121(i7+i8)

void fft13(T2 *u) {
  const double COS1 = 0.8854560256532098959003755220151;	// cos(tau/13)
  const double COS2 = 0.56806474673115580251180755912752;	// cos(2*tau/13)
  const double COS3 = 0.12053668025532305334906768745254;	// cos(3*tau/13)
  const double COS4 = -0.35460488704253562596963789260002;	// cos(4*tau/13)
  const double COS5 = -0.74851074817110109863463059970135;	// cos(5*tau/13)
  const double COS6 = -0.97094181742605202715698227629379;	// cos(6*tau/13)
  const double SIN1 = 0.4647231720437685456560153351331;	// sin(tau/13)
  const double SIN2_SIN1 = 1.7709120513064197918007510440302;	// sin(2*tau/13) / sin(tau/13) = .823/.465
  const double SIN3_SIN1 = 2.136129493462311605023615118255;	// sin(3*tau/13) / sin(tau/13) = .993/.465
  const double SIN4_SIN1 = 2.0119854118170658984988864189353;	// sin(4*tau/13) / sin(tau/13) = .935/.465
  const double SIN5_SIN1 = 1.426919719377240353084339333055;	// sin(5*tau/13) / sin(tau/13) = .663/.465
  const double SIN6_SIN1 = 0.51496391547486370122962521953258;	// sin(6*tau/13) / sin(tau/13) = .239/.465

  X2_mul_t4(u[1], u[12]);				// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[11]);				// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(u[3], u[10]);				// (r4+ i4+),  (i4- -r4-)
  X2_mul_t4(u[4], u[9]);				// (r5+ i5+),  (i5- -r5-)
  X2_mul_t4(u[5], u[8]);				// (r6+ i6+),  (i6- -r6-)
  X2_mul_t4(u[6], u[7]);				// (r7+ i7+),  (i7- -r7-)

  T2 tmp213a = fmaT2(COS1, u[1], u[0]);
  T2 tmp312a = fmaT2(COS2, u[1], u[0]);
  T2 tmp411a = fmaT2(COS3, u[1], u[0]);
  T2 tmp510a = fmaT2(COS4, u[1], u[0]);
  T2 tmp69a = fmaT2(COS5, u[1], u[0]);
  T2 tmp78a = fmaT2(COS6, u[1], u[0]);
  u[0] = u[0] + u[1];

  tmp213a = fmaT2(COS2, u[2], tmp213a);
  tmp312a = fmaT2(COS4, u[2], tmp312a);
  tmp411a = fmaT2(COS6, u[2], tmp411a);
  tmp510a = fmaT2(COS5, u[2], tmp510a);
  tmp69a = fmaT2(COS3, u[2], tmp69a);
  tmp78a = fmaT2(COS1, u[2], tmp78a);
  u[0] = u[0] + u[2];

  tmp213a = fmaT2(COS3, u[3], tmp213a);
  tmp312a = fmaT2(COS6, u[3], tmp312a);
  tmp411a = fmaT2(COS4, u[3], tmp411a);
  tmp510a = fmaT2(COS1, u[3], tmp510a);
  tmp69a = fmaT2(COS2, u[3], tmp69a);
  tmp78a = fmaT2(COS5, u[3], tmp78a);
  u[0] = u[0] + u[3];

  tmp213a = fmaT2(COS4, u[4], tmp213a);
  tmp312a = fmaT2(COS5, u[4], tmp312a);
  tmp411a = fmaT2(COS1, u[4], tmp411a);
  tmp510a = fmaT2(COS3, u[4], tmp510a);
  tmp69a = fmaT2(COS6, u[4], tmp69a);
  tmp78a = fmaT2(COS2, u[4], tmp78a);
  u[0] = u[0] + u[4];

  tmp213a = fmaT2(COS5, u[5], tmp213a);
  tmp312a = fmaT2(COS3, u[5], tmp312a);
  tmp411a = fmaT2(COS2, u[5], tmp411a);
  tmp510a = fmaT2(COS6, u[5], tmp510a);
  tmp69a = fmaT2(COS1, u[5], tmp69a);
  tmp78a = fmaT2(COS4, u[5], tmp78a);
  u[0] = u[0] + u[5];

  tmp213a = fmaT2(COS6, u[6], tmp213a);
  tmp312a = fmaT2(COS1, u[6], tmp312a);
  tmp411a = fmaT2(COS5, u[6], tmp411a);
  tmp510a = fmaT2(COS2, u[6], tmp510a);
  tmp69a = fmaT2(COS4, u[6], tmp69a);
  tmp78a = fmaT2(COS3, u[6], tmp78a);
  u[0] = u[0] + u[6];

  T2 tmp213b = fmaT2(SIN2_SIN1, u[11], u[12]);		// .823/.465
  T2 tmp312b = fmaT2(SIN2_SIN1, u[12], -u[7]);
  T2 tmp411b = fmaT2(SIN2_SIN1, u[8], -u[9]);
  T2 tmp510b = fmaT2(SIN2_SIN1, -u[7], -u[10]);
  T2 tmp69b = fmaT2(SIN2_SIN1, u[10], -u[8]);
  T2 tmp78b = fmaT2(SIN2_SIN1, -u[9], -u[11]);

  tmp213b = fmaT2(SIN3_SIN1, u[10], tmp213b);		// .993/.465
  tmp312b = fmaT2(SIN3_SIN1, -u[8], tmp312b);
  tmp411b = fmaT2(SIN3_SIN1, u[12], tmp411b);
  tmp510b = fmaT2(SIN3_SIN1, u[9], tmp510b);
  tmp69b = fmaT2(SIN3_SIN1, -u[11], tmp69b);
  tmp78b = fmaT2(SIN3_SIN1, -u[7], tmp78b);

  tmp213b = fmaT2(SIN4_SIN1, u[9], tmp213b);		// .935/.465
  tmp312b = fmaT2(SIN4_SIN1, u[11], tmp312b);
  tmp411b = fmaT2(SIN4_SIN1, -u[10], tmp411b);
  tmp510b = fmaT2(SIN4_SIN1, u[12], tmp510b);
  tmp69b = fmaT2(SIN4_SIN1, u[7], tmp69b);
  tmp78b = fmaT2(SIN4_SIN1, u[8], tmp78b);

  tmp213b = fmaT2(SIN5_SIN1, u[8], tmp213b);		// .663/.465
  tmp312b = fmaT2(SIN5_SIN1, -u[9], tmp312b);
  tmp411b = fmaT2(SIN5_SIN1, u[7], tmp411b);
  tmp510b = fmaT2(SIN5_SIN1, -u[11], tmp510b);
  tmp69b = fmaT2(SIN5_SIN1, u[12], tmp69b);
  tmp78b = fmaT2(SIN5_SIN1, u[10], tmp78b);

  tmp213b = fmaT2(SIN6_SIN1, u[7], tmp213b);		// .239/.465
  tmp312b = fmaT2(SIN6_SIN1, u[10], tmp312b);
  tmp411b = fmaT2(SIN6_SIN1, u[11], tmp411b);
  tmp510b = fmaT2(SIN6_SIN1, -u[8], tmp510b);
  tmp69b = fmaT2(SIN6_SIN1, -u[9], tmp69b);
  tmp78b = fmaT2(SIN6_SIN1, u[12], tmp78b);

  fma_addsub(u[1], u[12], SIN1, tmp213a, tmp213b);
  fma_addsub(u[2], u[11], SIN1, tmp312a, tmp312b);
  fma_addsub(u[3], u[10], SIN1, tmp411a, tmp411b);
  fma_addsub(u[4], u[9], SIN1, tmp510a, tmp510b);
  fma_addsub(u[5], u[8], SIN1, tmp69a, tmp69b);
  fma_addsub(u[6], u[7], SIN1, tmp78a, tmp78b);
}
)cltag",

// src/cl/fft14.cl
R"cltag(
// Copyright (C) Mihai Preda and George Woltman

void fft14(T2 *u) {
  const double SIN1 = 0.781831482468029809;		// sin(tau/7)
  const double SIN2_SIN1 = 1.2469796037174670611;	// sin(2*tau/7) / sin(tau/7) = .975/.782
  const double SIN3_SIN1 = 0.5549581320873711914;	// sin(3*tau/7) / sin(tau/7) = .434/.782
  const double COS1 = 0.6234898018587335305;		// cos(tau/7)
  const double COS2 = -0.2225209339563144043;		// cos(2*tau/7)
  const double COS3 = -0.9009688679024191262;		// cos(3*tau/7)

  X2(u[0], u[7]);					// (r1+ i1+),  (r1-  i1-)
  X2_mul_t4(u[1], u[8]);				// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[9]);				// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(u[3], u[10]);				// (r4+ i4+),  (i4- -r4-)
  X2_mul_t4(u[4], u[11]);				// (r5+ i5+),  (i5- -r5-)
  X2_mul_t4(u[5], u[12]);				// (r6+ i6+),  (i6- -r6-)
  X2_mul_t4(u[6], u[13]);				// (r7+ i7+),  (i7- -r7-)

  X2_mul_t4(u[1], u[6]);				// (r2++  i2++),  (i2+- -r2+-)
  X2_mul_t4(u[2], u[5]);				// (r3++  i3++),  (i3+- -r3+-)
  X2_mul_t4(u[3], u[4]);				// (r4++  i4++),  (i4+- -r4+-)
  X2_mul_t4(u[8], u[13]);				// (i2-+ -r2-+), (-r2-- -i2--)
  X2_mul_t4(u[9], u[12]);				// (i3-+ -r3-+), (-r3-- -i3--)
  X2_mul_t4(u[10], u[11]);				// (i4-+ -r4-+), (-r4-- -i4--)

  T2 tmp313a = fmaT2(COS1, u[1], u[0]);
  T2 tmp511a = fmaT2(COS2, u[1], u[0]);
  T2 tmp79a = fmaT2(COS3, u[1], u[0]);
  u[0] = u[0] + u[1];
  T2 tmp214a = fmaT2(COS3, u[13], u[7]);
  T2 tmp412a = fmaT2(COS2, u[13], u[7]);
  T2 tmp610a = fmaT2(COS1, u[13], u[7]);
  u[7] = u[7] + u[13];

  tmp313a = fmaT2(COS2, u[2], tmp313a);
  tmp511a = fmaT2(COS3, u[2], tmp511a);
  tmp79a = fmaT2(COS1, u[2], tmp79a);
  u[0] = u[0] + u[2];
  tmp214a = fmaT2(COS1, -u[12], tmp214a);
  tmp412a = fmaT2(COS3, -u[12], tmp412a);
  tmp610a = fmaT2(COS2, -u[12], tmp610a);
  u[7] = u[7] - u[12];

  tmp313a = fmaT2(COS3, u[3], tmp313a);
  tmp511a = fmaT2(COS1, u[3], tmp511a);
  tmp79a = fmaT2(COS2, u[3], tmp79a);
  u[0] = u[0] + u[3];
  tmp214a = fmaT2(COS2, u[11], tmp214a);
  tmp412a = fmaT2(COS1, u[11], tmp412a);
  tmp610a = fmaT2(COS3, u[11], tmp610a);
  u[7] = u[7] + u[11];

  T2 tmp313b = fmaT2(SIN2_SIN1, u[5], u[6]);			// Apply .975/.782
  T2 tmp511b = fmaT2(SIN2_SIN1, u[6], -u[4]);
  T2 tmp79b = fmaT2(SIN2_SIN1, u[4], -u[5]);
  T2 tmp214b = fmaT2(SIN2_SIN1, u[10], u[9]);
  T2 tmp412b = fmaT2(SIN2_SIN1, u[8], -u[10]);
  T2 tmp610b = fmaT2(SIN2_SIN1, -u[9], u[8]);

  tmp313b = fmaT2(SIN3_SIN1, u[4], tmp313b);			// Apply .434/.782
  tmp511b = fmaT2(SIN3_SIN1, -u[5], tmp511b);
  tmp79b = fmaT2(SIN3_SIN1, u[6], tmp79b);
  tmp214b = fmaT2(SIN3_SIN1, u[8], tmp214b);
  tmp412b = fmaT2(SIN3_SIN1, u[9], tmp412b);
  tmp610b = fmaT2(SIN3_SIN1, u[10], tmp610b);

  fma_addsub(u[1], u[13], SIN1, tmp214a, tmp214b);
  fma_addsub(u[2], u[12], SIN1, tmp313a, tmp313b);
  fma_addsub(u[3], u[11], SIN1, tmp412a, tmp412b);
  fma_addsub(u[4], u[10], SIN1, tmp511a, tmp511b);
  fma_addsub(u[5], u[9], SIN1, tmp610a, tmp610b);
  fma_addsub(u[6], u[8], SIN1, tmp79a, tmp79b);
}
)cltag",

// src/cl/fft15.cl
R"cltag(
// Copyright (C) Mihai Preda and George Woltman

#include "fft3.cl"
#include "fft5.cl"

// 5 complex FFT where second though fifth inputs need to be multiplied by SIN1, and third input needs to multiplied by SIN2
void fft5delayedSIN1234(T2 *u) {
  const double SIN4_SIN1 = 2.44512490403509663921;		// sin(4*tau/15) / sin(tau/15) = .985/.643
  const double SIN3_SIN2 = 1.27977277603217842055;		// sin(3*tau/15) / sin(2*tau/15) = .985/.643
  const double COS1SIN1 = 0.12568853494543955095;		// cos(tau/5) * sin(tau/15) = .309 * .407
  const double COS1SIN2 = 0.2296443803543192195;		// cos(tau/5) * sin(2*tau/15) = .309 * .743
  const double COS2SIN1 = -0.32905685648333965483;		// cos(2*tau/5) * sin(tau/15) = -.809 * .407
  const double COS2SIN2 = -0.60121679309301633701;		// cos(2*tau/5) * sin(2*tau/15) = -.809 * .743
  const double SIN1 = 0.40673664307580020775;			// sin(tau/15) = .407
  const double SIN2 = 0.74314482547739423501;			// sin(2*tau/15) = .743
  const double SIN2_SIN1SIN1_SIN2 = 0.33826121271771642765;	// sin(2*tau/5) / sin(tau/5) * sin(tau/15) / sin(2*tau/15) = .588/.951 * .407/.743
  const double SIN2_SIN1SIN2_SIN1 = 1.12920428618240948485;	// sin(2*tau/5) / sin(tau/5) * sin(2*tau/15) / sin(tau/15) = .588/.951 * .743/.407
  const double SIN1SIN1 = 0.38682953481325584261;		// sin(tau/5) * sin(tau/15) = .951 * .407
  const double SIN1SIN2 = 0.70677272882130044775;		// sin(tau/5) * sin(2*tau/15) = .951 * .743

  fma_addsub(u[1], u[4], SIN4_SIN1, u[1], u[4]);		// (r2+ i2+),  (i2- -r2-)		we owe results a mul by SIN1
  u[4] = mul_t4(u[4]);
  fma_addsub(u[2], u[3], SIN3_SIN2, u[2], u[3]);		// (r3+ i3+),  (i3- -r3-)		we owe results a mul by SIN2
  u[3] = mul_t4(u[3]);

  T2 tmp25a = fmaT2(COS1SIN1, u[1], u[0]);
  T2 tmp34a = fmaT2(COS2SIN1, u[1], u[0]);
  u[0] = u[0] + SIN1 * u[1];

  tmp25a = fmaT2(COS2SIN2, u[2], tmp25a);
  tmp34a = fmaT2(COS1SIN2, u[2], tmp34a);
  u[0] = u[0] + SIN2 * u[2];

  T2 tmp25b = fmaT2(SIN2_SIN1SIN2_SIN1, u[3], u[4]);		// (i2- +.588/.951*i3-, -r2- -.588/.951*r3-)	we owe results a mul by .951*SIN1
  T2 tmp34b = fmaT2(SIN2_SIN1SIN1_SIN2, u[4], -u[3]);		// (.588/.951*i2- -i3-, -.588/.951*r2- +r3-)	we owe results a mul by .951*SIN2

  fma_addsub(u[1], u[4], SIN1SIN1, tmp25a, tmp25b);
  fma_addsub(u[2], u[3], SIN1SIN2, tmp34a, tmp34b);
}

// This version is faster (fewer F64 ops), but slightly less accurate
void fft15(T2 *u) {
  const double COS1_SIN1 = 2.24603677390421605416;	// cos(tau/15) / sin(tau/15) = .766/.643
  const double COS2_SIN2 = 0.90040404429783994512;	// cos(2*tau/15) / sin(2*tau/15) = .174/.985
  const double COS3_SIN3 = 0.32491969623290632616;	// cos(3*tau/15) / sin(3*tau/15) = .174/.985
  const double COS4_SIN4 = -0.10510423526567646251;	// cos(4*tau/15) / sin(4*tau/15) = .174/.985

  fft3by(u, 5);
  fft3by(u+1, 5);
  fft3by(u+2, 5);
  fft3by(u+3, 5);
  fft3by(u+4, 5);

  u[6] = partial_cmul(u[6], COS1_SIN1);			// mul by w^1, we owe result a mul by SIN1
  u[11] = partial_cmul_conjugate(u[11], COS1_SIN1);	// mul by w^-1, we owe result a mul by SIN1
  u[7] = partial_cmul(u[7], COS2_SIN2);			// mul by w^2, we owe result a mul by SIN2
  u[12] = partial_cmul_conjugate(u[12], COS2_SIN2);	// mul by w^-2, we owe result a mul by SIN2
  u[8] = partial_cmul(u[8], COS3_SIN3);			// mul by w^3, we owe result a mul by SIN3
  u[13] = partial_cmul_conjugate(u[13], COS3_SIN3);	// mul by w^-3, we owe result a mul by SIN3
  u[9] = partial_cmul(u[9], COS4_SIN4);			// mul by w^4, we owe result a mul by SIN4
  u[14] = partial_cmul_conjugate(u[14], COS4_SIN4);	// mul by w^-4, we owe result a mul by SIN4

  fft5(u);
  fft5delayedSIN1234(u+5);
  fft5delayedSIN1234(u+10);

  // fix order [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 14, 2, 5, 8, 11]

  T2 tmp = u[1];
  u[1] = u[5];
  u[5] = u[12];
  u[12] = u[4];
  u[4] = u[6];
  u[6] = u[2];
  u[2] = u[11];
  u[11] = u[14];
  u[14] = u[10];
  u[10] = u[8];
  u[8] = u[13];
  u[13] = u[9];
  u[9] = u[3];
  u[3] = tmp;
}
)cltag",

// src/cl/fft3.cl
R"cltag(
// Copyright (C) Mihai Preda

void fft3by(T2 *u, u32 incr) {
  const double COS1 = -0.5;					// cos(tau/3), -0.5
  const double SIN1 = 0.86602540378443864676372317075294;	// sin(tau/3), sqrt(3)/2, 0.86602540378443864676372317075294
  X2_mul_t4(u[1*incr], u[2*incr]);				// (r2+r3 i2+i3),  (i2-i3 -(r2-r3))
  T2 tmp23 = u[0*incr] + COS1 * u[1*incr];
  u[0*incr] = u[0*incr] + u[1*incr];
  fma_addsub(u[1*incr], u[2*incr], SIN1, tmp23, u[2*incr]);
}

void fft3(T2 *u) {
  fft3by(u, 1);
}
)cltag",

// src/cl/fft4.cl
R"cltag(
// Copyright (C) Mihai Preda

#pragma once

void fft4Core(T2 *u) {
  X2(u[0], u[2]);
  X2(u[1], u[3]);
  X2(u[0], u[1]);

  T t = u[3].x;
  u[3].x = u[2].x - u[3].y;
  u[2].x = u[2].x + u[3].y;
  u[3].y = u[2].y + t;
  u[2].y = u[2].y - t;
}

void fft4(T2 *u) {
   fft4Core(u);
   // revbin [0 2 1 3] undo
   SWAP(u[1], u[2]);
}
)cltag",

// src/cl/fft5.cl
R"cltag(
// Copyright (C) Mihai Preda and George Woltman

#if !NEWEST_FFT5 && !NEW_FFT5 && !OLD_FFT5
#define NEW_FFT5 1
#endif

// Adapted from: Nussbaumer, "Fast Fourier Transform and Convolution Algorithms", 5.5.4 "5-Point DFT".

// Using rocm 2.9, testKernel shows this macro generates 38 f64 (8 FMA) ops, 26 vgprs.
#if OLD_FFT5
void fft5(T2 *u) {
  const double SIN1 = 0x1.e6f0e134454ffp-1; // sin(tau/5), 0.95105651629515353118
  const double SIN2 = 0x1.89f188bdcd7afp+0; // sin(tau/5) + sin(2*tau/5), 1.53884176858762677931
  const double SIN3 = 0x1.73fd61d9df543p-2; // sin(tau/5) - sin(2*tau/5), 0.36327126400268044959
  const double COS1 = 0x1.1e3779b97f4a8p-1; // (cos(tau/5) - cos(2*tau/5))/2, 0.55901699437494745126

  X2(u[2], u[3]);
  X2(u[1], u[4]);
  X2(u[1], u[2]);

  T2 tmp = u[0];
  u[0] += u[1];
  u[1] = u[1] * (-0.25) + tmp;

  u[2] *= COS1;

  tmp = (u[4] - u[3]) * SIN1;
  tmp  = U2(tmp.y, -tmp.x);

  u[3] = U2(u[3].y, -u[3].x) * SIN2 + tmp;
  u[4] = U2(-u[4].y, u[4].x) * SIN3 + tmp;
  SWAP(u[3], u[4]);

  X2(u[1], u[2]);
  X2(u[1], u[4]);
  X2(u[2], u[3]);
}

// Using rocm 2.9, testKernel shows this macro generates an ideal 44 f64 ops (12 FMA) or 32 f64 ops (20 FMA), 30 vgprs.
#elif NEW_FFT5

// Above uses fewer FMAs.  Above may be faster if FMA latency cannot be masked.
// Nussbaumer's ideas can be used to reduce FMAs -- see NEWEST_FFT5 implementation below.
// See prime95's gwnum/zr5.mac file for more detailed explanation of the formulas below
// R1= r1     +(r2+r5)     +(r3+r4)
// R2= r1 +.309(r2+r5) -.809(r3+r4)    +.951(i2-i5) +.588(i3-i4)
// R5= r1 +.309(r2+r5) -.809(r3+r4)    -.951(i2-i5) -.588(i3-i4)
// R3= r1 -.809(r2+r5) +.309(r3+r4)    +.588(i2-i5) -.951(i3-i4)
// R4= r1 -.809(r2+r5) +.309(r3+r4)    -.588(i2-i5) +.951(i3-i4)
// I1= i1     +(i2+i5)     +(i3+i4)
// I2= i1 +.309(i2+i5) -.809(i3+i4)    -.951(r2-r5) -.588(r3-r4)
// I5= i1 +.309(i2+i5) -.809(i3+i4)    +.951(r2-r5) +.588(r3-r4)
// I3= i1 -.809(i2+i5) +.309(i3+i4)    -.588(r2-r5) +.951(r3-r4)
// I4= i1 -.809(i2+i5) +.309(i3+i4)    +.588(r2-r5) -.951(r3-r4)

void fft5(T2 *u) {
  const double SIN1 = 0x1.e6f0e134454ffp-1;		// sin(tau/5), 0.95105651629515353118
  const double SIN2_SIN1 = 0.618033988749894848;	// sin(2*tau/5) / sin(tau/5) = .588/.951, 0.618033988749894848
  const double COS1 = 0.309016994374947424;		// cos(tau/5), 0.309016994374947424
  const double COS2 = 0.809016994374947424;		// -cos(2*tau/5), 0.809016994374947424

  X2_mul_t4(u[1], u[4]);				// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[3]);				// (r3+ i3+),  (i3- -r3-)

  T2 tmp25a = fmaT2(COS1, u[1], u[0]);
  T2 tmp34a = fmaT2(-COS2, u[1], u[0]);
  u[0] = u[0] + u[1];

  T2 tmp25b = fmaT2(SIN2_SIN1, u[3], u[4]);		// (i2- +.588/.951*i3-, -r2- -.588/.951*r3-)
  T2 tmp34b = fmaT2(SIN2_SIN1, u[4], -u[3]);		// (.588/.951*i2- -i3-, -.588/.951*r2- +r3-)

  tmp25a = fmaT2(-COS2, u[2], tmp25a);
  tmp34a = fmaT2(COS1, u[2], tmp34a);
  u[0] = u[0] + u[2];

  fma_addsub(u[1], u[4], SIN1, tmp25a, tmp25b);
  fma_addsub(u[2], u[3], SIN1, tmp34a, tmp34b);
}

// Using rocm 2.9, testKernel shows this macro generates an ideal 44 f64 ops (12 FMA) or 32 f64 ops (20 FMA), 30 vgprs.
#elif NEWEST_FFT5

// Nussbaumer's ideas used to introduce more PREFER_NOFMA opportunities in the code below.
// Modified prime95's formulas:
// R1= r1 + ((r2+r5)+(r3+r4))
// R2= r1 - ((r2+r5)+(r3+r4))/4 +.559((r2+r5)-(r3+r4))    +.951(i2-i5) +.588(i3-i4)
// R5= r1 - ((r2+r5)+(r3+r4))/4 +.559((r2+r5)-(r3+r4))    -.951(i2-i5) -.588(i3-i4)
// R3= r1 - ((r2+r5)+(r3+r4))/4 -.559((r2+r5)-(r3+r4))    +.588(i2-i5) -.951(i3-i4)
// R4= r1 - ((r2+r5)+(r3+r4))/4 -.559((r2+r5)-(r3+r4))    -.588(i2-i5) +.951(i3-i4)
// I1= i1 + ((i2+i5)+(i3+i4))
// I2= i1 - ((i2+i5)+(i3+i4))/4 +.559((i2+i5)-(i3+i4))    -.951(r2-r5) -.588(r3-r4)
// I5= i1 - ((i2+i5)+(i3+i4))/4 +.559((i2+i5)-(i3+i4))    +.951(r2-r5) +.588(r3-r4)
// I3= i1 - ((i2+i5)+(i3+i4))/4 -.559((i2+i5)-(i3+i4))    -.588(r2-r5) +.951(r3-r4)
// I4= i1 - ((i2+i5)+(i3+i4))/4 -.559((i2+i5)-(i3+i4))    +.588(r2-r5) -.951(r3-r4)

void fft5(T2 *u) {
  const double SIN1 = 0x1.e6f0e134454ffp-1;		// sin(tau/5), 0.95105651629515353118
  const double SIN2_SIN1 = 0.618033988749894848;	// sin(2*tau/5) / sin(tau/5) = .588/.951, 0.618033988749894848
  const double COS12 = 0x1.1e3779b97f4a8p-1;		// (cos(tau/5) - cos(2*tau/5))/2, 0.55901699437494745126

  X2_mul_t4(u[1], u[4]);				// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[3]);				// (r3+ i3+),  (i3- -r3-)
  X2(u[1], u[2]);					// (r2++ i2++), (r2+- i2+-)

  T2 tmp2345a = fmaT2(-0.25, u[1], u[0]);
  u[0] = u[0] + u[1];

  T2 tmp25b = fmaT2(SIN2_SIN1, u[3], u[4]);		// (i2- +.588/.951*i3-, -r2- -.588/.951*r3-)
  T2 tmp34b = fmaT2(SIN2_SIN1, u[4], -u[3]);		// (.588/.951*i2- -i3-, -.588/.951*r2- +r3-)

  T2 tmp25a, tmp34a;
  fma_addsub(tmp25a, tmp34a, COS12, tmp2345a, u[2]);

  fma_addsub(u[1], u[4], SIN1, tmp25a, tmp25b);
  fma_addsub(u[2], u[3], SIN1, tmp34a, tmp34b);
}
#else
#error None of OLD_FFT5, NEW_FFT5, NEWEST_FFT5 defined
#endif
)cltag",

// src/cl/fft6.cl
R"cltag(
void fft6(T2 *u) {
  const double COS1 = -0.5;					                  // cos(tau/3) == -0.5
  const double SIN1 = 0.86602540378443864676372317075294;	// sin(tau/3) == sqrt(3)/2

  X2(u[0], u[3]);						// (r1+ i1+),  (r1-  i1-)
  X2_mul_t4(u[1], u[4]);					// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[5]);					// (r3+ i3+),  (i3- -r3-)

  X2_mul_t4(u[1], u[2]);					// (r2++  i2++),  (i2+- -r2+-)
  X2_mul_t4(u[4], u[5]);					// (i2-+ -r2-+), (-r2-- -i2--)

  T2 tmp35a = fmaT2(COS1, u[1], u[0]);
  u[0] = u[0] + u[1];
  T2 tmp26a = fmaT2(COS1, u[5], u[3]);
  u[3] = u[3] + u[5];

  fma_addsub(u[1], u[5], SIN1, tmp26a, u[4]);
  fma_addsub(u[2], u[4], SIN1, tmp35a, u[2]);
}
)cltag",

// src/cl/fft7.cl
R"cltag(

// See prime95's gwnum/zr7.mac file for more detailed explanation of the formulas below
// R1= r1     +(r2+r7)     +(r3+r6)     +(r4+r5)
// R2= r1 +.623(r2+r7) -.223(r3+r6) -.901(r4+r5)  +(.782(i2-i7) +.975(i3-i6) +.434(i4-i5))
// R7= r1 +.623(r2+r7) -.223(r3+r6) -.901(r4+r5)  -(.782(i2-i7) +.975(i3-i6) +.434(i4-i5))
// R3= r1 -.223(r2+r7) -.901(r3+r6) +.623(r4+r5)  +(.975(i2-i7) -.434(i3-i6) -.782(i4-i5))
// R6= r1 -.223(r2+r7) -.901(r3+r6) +.623(r4+r5)  -(.975(i2-i7) -.434(i3-i6) -.782(i4-i5))
// R4= r1 -.901(r2+r7) +.623(r3+r6) -.223(r4+r5)  +(.434(i2-i7) -.782(i3-i6) +.975(i4-i5))
// R5= r1 -.901(r2+r7) +.623(r3+r6) -.223(r4+r5)  -(.434(i2-i7) -.782(i3-i6) +.975(i4-i5))

// I1= i1     +(i2+i7)     +(i3+i6)     +(i4+i5)
// I2= i1 +.623(i2+i7) -.223(i3+i6) -.901(i4+i5)  -(.782(r2-r7) +.975(r3-r6) +.434(r4-r5))
// I7= i1 +.623(i2+i7) -.223(i3+i6) -.901(i4+i5)  +(.782(r2-r7) +.975(r3-r6) +.434(r4-r5))
// I3= i1 -.223(i2+i7) -.901(i3+i6) +.623(i4+i5)  -(.975(r2-r7) -.434(r3-r6) -.782(r4-r5))
// I6= i1 -.223(i2+i7) -.901(i3+i6) +.623(i4+i5)  +(.975(r2-r7) -.434(r3-r6) -.782(r4-r5))
// I4= i1 -.901(i2+i7) +.623(i3+i6) -.223(i4+i5)  -(.434(r2-r7) -.782(r3-r6) +.975(r4-r5))
// I5= i1 -.901(i2+i7) +.623(i3+i6) -.223(i4+i5)  +(.434(r2-r7) -.782(r3-r6) +.975(r4-r5))

void fft7(T2 *u) {
  const double COS1 = 0.6234898018587335305;		// cos(tau/7)
  const double COS2 = -0.2225209339563144043;		// cos(2*tau/7)
  const double COS3 = -0.9009688679024191262;		// cos(3*tau/7)
  const double SIN1 = 0.781831482468029809;		// sin(tau/7)
  const double SIN2_SIN1 = 1.2469796037174670611;	// sin(2*tau/7) / sin(tau/7) = .975/.782
  const double SIN3_SIN1 = 0.5549581320873711914;	// sin(3*tau/7) / sin(tau/7) = .434/.782

  X2_mul_t4(u[1], u[6]);				// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[5]);				// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(u[3], u[4]);				// (r4+ i4+),  (i4- -r4-)

  T2 tmp27a = fmaT2(COS1, u[1], u[0]);
  T2 tmp36a = fmaT2(COS2, u[1], u[0]);
  T2 tmp45a = fmaT2(COS3, u[1], u[0]);
  u[0] = u[0] + u[1];

  tmp27a = fmaT2(COS2, u[2], tmp27a);
  tmp36a = fmaT2(COS3, u[2], tmp36a);
  tmp45a = fmaT2(COS1, u[2], tmp45a);
  u[0] = u[0] + u[2];

  tmp27a = fmaT2(COS3, u[3], tmp27a);
  tmp36a = fmaT2(COS1, u[3], tmp36a);
  tmp45a = fmaT2(COS2, u[3], tmp45a);
  u[0] = u[0] + u[3];

  T2 tmp27b = fmaT2(SIN2_SIN1, u[5], u[6]);		// .975/.782
  T2 tmp36b = fmaT2(SIN2_SIN1, u[6], -u[4]);
  T2 tmp45b = fmaT2(SIN2_SIN1, u[4], -u[5]);

  tmp27b = fmaT2(SIN3_SIN1, u[4], tmp27b);		// .434/.782
  tmp36b = fmaT2(SIN3_SIN1, -u[5], tmp36b);
  tmp45b = fmaT2(SIN3_SIN1, u[6], tmp45b);

  fma_addsub(u[1], u[6], SIN1, tmp27a, tmp27b);
  fma_addsub(u[2], u[5], SIN1, tmp36a, tmp36b);
  fma_addsub(u[3], u[4], SIN1, tmp45a, tmp45b);
}
)cltag",

// src/cl/fft8.cl
R"cltag(
// Copyright (C) Mihai Preda

#pragma once

#include "fft4.cl"

void fft8Core(T2 *u) {
  X2(u[0], u[4]);
  X2(u[1], u[5]);   u[5] = mul_t8(u[5]);
  X2(u[2], u[6]);   u[6] = mul_t4(u[6]);
  X2(u[3], u[7]);   u[7] = mul_3t8(u[7]);
  fft4Core(u);
  fft4Core(u + 4);
}

void fft8(T2 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}
)cltag",

// src/cl/fft9.cl
R"cltag(
// Copyright (C) Mihai Preda and George Woltman

#include "fft3.cl"

#if !NEW_FFT9 && !OLD_FFT9
#define NEW_FFT9 1
#endif

#if OLD_FFT9
// Adapted from: Nussbaumer, "Fast Fourier Transform and Convolution Algorithms", 5.5.7 "9-Point DFT".
void fft9(T2 *u) {
  const double C0 = 0x1.8836fa2cf5039p-1; //   0.766044443118978013 (2*c(u) - c(2*u) - c(4*u))/3
  const double C1 = 0x1.e11f642522d1cp-1; //   0.939692620785908428 (c(u) + c(2*u) - 2*c(4*u))/3
  const double C2 = 0x1.63a1a7e0b738ap-3; //   0.173648177666930359 -(c(u) - 2*c(2*u) + c(4*u))/3
  const double C3 = 0x1.bb67ae8584caap-1; //   0.866025403784438597 s(3*u)
  const double C4 = 0x1.491b7523c161dp-1; //   0.642787609686539363 s(u)
  const double C5 = 0x1.5e3a8748a0bf5p-2; //   0.342020143325668713 s(4*u)
  const double C6 = 0x1.f838b8c811c17p-1; //   0.984807753012208020 s(2*u)

  X2(u[1], u[8]);
  X2(u[2], u[7]);
  X2(u[3], u[6]);
  X2(u[4], u[5]);

  T2 m4 = (u[2] - u[4]) * C1;
  T2 s0 = (u[2] - u[1]) * C0 - m4;

  X2(u[1], u[4]);

  T2 t5 = u[1] + u[2];

  T2 m8  = mul_t4(u[7] + u[8]) * C4;
  T2 m10 = mul_t4(u[5] - u[8]) * C6;

  X2(u[5], u[7]);

  T2 m9  = mul_t4(u[5]) * C5;
  T2 t10 = u[8] + u[7];

  T2 s2 = m8 + m9;
  u[5] = m9 - m10;

  u[2] = u[0] - u[3] / 2;
  u[0] += u[3];
  u[3] = u[0] - t5 / 2;
  u[0] += t5;

  u[7] = mul_t4(u[6]) * C3;
  u[8] = u[7] + s2;
  u[6] = mul_t4(t10)  * C3;

  u[1] = u[2] - s0;

  u[4] = u[4] * C2 - m4;

  X2(u[2], u[4]);

  u[4] += s0;

  X2(u[5], u[7]);
  u[5] -= s2;

  X2(u[4], u[5]);
  X2(u[3], u[6]);
  X2(u[2], u[7]);
  X2(u[1], u[8]);
}

#elif NEW_FFT9

// 3 complex FFT where second input needs to be multiplied by SIN1 and third input needs to multiplied by SIN2
void fft3delayedSIN12(T2 *u) {
  const double SIN2_SIN1 = 1.5320888862379560704047853011108;	// sin(2*tau/9) / sin(tau/9) = .985/.643
  const double COS1SIN1 = -0.32139380484326966316132170495363;	// cos(tau/3) * sin(tau/9) = -.5 * .643
  const double SIN1SIN1 = 0.55667039922641936645291295204702;	// sin(tau/3) * sin(tau/9) = .866 * .643
  const double SIN1 = 0.64278760968653932632264340990726;	// sin(tau/9) = .643
  fma_addsub(u[1], u[2], SIN2_SIN1, u[1], u[2]);		// (r2+r3 i2+i3),  (i2-i3 -(r2-r3))	we owe results a mul by SIN1
  u[2] = mul_t4(u[2]);
  T2 tmp23 = u[0] + COS1SIN1 * u[1];
  u[0] = u[0] + SIN1 * u[1];
  fma_addsub (u[1], u[2], SIN1SIN1, tmp23, u[2]);
}

// This version is faster (fewer F64 ops), but slightly less accurate
void fft9(T2 *u) {
  const double COS1_SIN1 = 1.1917535925942099587053080718604;	// cos(tau/9) / sin(tau/9) = .766/.643
  const double COS2_SIN2 = 0.17632698070846497347109038686862;	// cos(2*tau/9) / sin(2*tau/9) = .174/.985

  fft3by(u, 3);
  fft3by(u+1, 3);
  fft3by(u+2, 3);

  u[4] = partial_cmul(u[4], COS1_SIN1);			// mul u[4] by w^1, we owe result a mul by SIN1
  u[7] = partial_cmul_conjugate(u[7], COS1_SIN1);	// mul u[7] by w^-1, we owe result a mul by SIN1
  u[5] = partial_cmul(u[5], COS2_SIN2);			// mul u[5] by w^2, we owe result a mul by SIN2
  u[8] = partial_cmul_conjugate(u[8], COS2_SIN2);	// mul u[8] by w^-2, we owe result a mul by SIN2

  fft3(u);
  fft3delayedSIN12(u+3);
  fft3delayedSIN12(u+6);

  // fix order [0, 3, 6, 1, 4, 7, 8, 2, 5]

  T2 tmp = u[1];
  u[1] = u[3];
  u[3] = tmp;
  tmp = u[2];
  u[2] = u[7];
  u[7] = u[5];
  u[5] = u[8];
  u[8] = u[6];
  u[6] = tmp;
}
#endif
)cltag",

// src/cl/fftbase.cl
R"cltag(
// Copyright (C) Mihai Preda

#include "fft4.cl"
#include "fft8.cl"
#include "trig.cl"
// #include "math.cl"

void shufl(u32 WG, local T2 *lds2, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  local T* lds = (local T*) lds2;

  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);

  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i].x; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].x = lds[i * WG + me]; }
  bar();
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i].y; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].y = lds[i * WG + me]; }
}

void tabMul(u32 WG, Trig trig, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  u32 p = me & ~(f - 1);
  T2 w = trig[p];

  if (n >= 8) {
    u[1] = fancyMulTrig(u[1], w);
  } else {
    u[1] = mul(u[1], w);
  }

#if CLEAN == 1
  T2 base = trig[WG + p];

  if (n >= 8) {
    for (u32 i = 2; i < n; ++i) {
      u[i] = mul(u[i], base);
      base = fancyMulTrig(base, w);
    }
  } else {
    for (u32 i = 2; i < n; ++i) {
      u[i] = mul(u[i], base);
      base = mul(base, w);
    }
  }

#elif CLEAN == 0
  if (n >= 8) {
    T a = 2 * fma(w.x, w.y, w.y); // 2*sin*cos
    u[2] = fancyMulTrig(u[2], U2(-2 * w.y * w.y, a));
    a *= 2;
    T2 base = U2(fma(a, -w.y, w.x + 1), fma(a, w.x, a - w.y));
    for (u32 i = 3; i < n; ++i) {
      u[i] = mul(u[i], base);
      base = fancyMulTrig(base, w);
    }
  } else {
    T a = 2 * w.x * w.y;
    // u[2] = fancyMulTrig(u[2], U2(-2 * w.y * w.y, a));
    // u[2] = mul(u[2], U2(fma(w.x, w.x, -w.y * w.y), a));
    u[2] = mul(u[2], U2(fma(-2 * w.y, w.y, 1), a));
    a *= 2;
    T2 base = U2(fma(a, -w.y, w.x), fma(a, w.x, -w.y));
    for (u32 i = 3; i < n; ++i) {
      u[i] = mul(u[i], base);
      base = mul(base, w);
    }
  }
#else
#error CLEAN must be 0 or 1
#endif
}

void shuflAndMul(u32 WG, local T2 *lds, Trig trig, T2 *u, u32 n, u32 f) {
  tabMul(WG, trig, u, n, f);
  shufl(WG, lds, u, n, f);
}
)cltag",

// src/cl/fftheight.cl
R"cltag(
// Copyright (C) Mihai Preda

#include "base.cl"
#include "fftbase.cl"
#include "middle.cl"

u32 transPos(u32 k, u32 middle, u32 width) { return k / width + k % width * middle; }

void fft256h(local T2 *lds, T2 *u, Trig trig) {
  for (u32 s = 0; s <= 4; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul(64, lds, trig, u, 4, 1u << s);
  }
  fft4(u);
}

void fft512h(local T2 *lds, T2 *u, Trig trig) {
  for (u32 s = 0; s <= 3; s += 3) {
    if (s) { bar(); }
    fft8(u);
    shuflAndMul(64, lds, trig, u, 8, 1u << s);
  }
  fft8(u);
}

void fft1Kh(local T2 *lds, T2 *u, Trig trig) {
  for (i32 s = 0; s <= 6; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul(256, lds, trig, u, 4, 1u << s);
  }
  fft4(u);
}

void fft_NH(T2 *u) {
#if NH == 4
  fft4(u);
#elif NH == 8
  fft8(u);
#else
#error NH
#endif
}

void fft4Kh(local T2 *lds, T2 *u, Trig trig) {
  for (u32 s = 1; s < SMALL_HEIGHT / NH; s *= NH) {
    if (s > 1) { bar(); }
    fft_NH(u);
    shuflAndMul(SMALL_HEIGHT / NH, lds, trig, u, NH, s);
  }
  fft_NH(u);
}

void fft_HEIGHT(local T2 *lds, T2 *u, Trig trig) {
#if SMALL_HEIGHT == 256
  fft256h(lds, u, trig);
#elif SMALL_HEIGHT == 512
  fft512h(lds, u, trig);
#elif SMALL_HEIGHT == 1024
  fft1Kh(lds, u, trig);
#elif SMALL_HEIGHT == 4096
  fft4Kh(lds, u, trig);
#else
#error unexpected SMALL_HEIGHT.
#endif
}
)cltag",

// src/cl/ffthin.cl
R"cltag(
// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"
#include "fftheight.cl"

// Do an FFT Height after a transposeW (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHin(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[SMALL_HEIGHT / 2];
  
  T2 u[NH];
  u32 g = get_group_id(0);

  readTailFusedLine(in, u, g);
  fft_HEIGHT(lds, u, smallTrig);

  write(G_H, NH, u, out, SMALL_HEIGHT * transPos(g, MIDDLE, WIDTH));
}
)cltag",

// src/cl/fftmiddlein.cl
R"cltag(
// Copyright (C) Mihai Preda and George Woltman

#include "base.cl"
#include "math.cl"
#include "fft-middle.cl"
#include "middle.cl"

KERNEL(IN_WG) fftMiddleIn(P(T2) out, CP(T2) in, Trig trig, BigTab TRIG_BHW) {
  T2 u[MIDDLE];
  
  u32 SIZEY = IN_WG / IN_SIZEX;

  u32 N = WIDTH / IN_SIZEX;
  
  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % IN_SIZEX;
  u32 my = me / IN_SIZEX;

  u32 startx = gx * IN_SIZEX;
  u32 starty = gy * SIZEY;

  in += starty * WIDTH + startx;
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = in[i * SMALL_HEIGHT * WIDTH + my * WIDTH + mx]; }

  middleMul2(u, startx + mx, starty + my, 1, trig, TRIG_BHW);

  fft_MIDDLE(u);

  middleMul(u, starty + my, trig, TRIG_BHW);

  local T lds[IN_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);

  write(IN_WG, MIDDLE, u, out, gx * (BIG_HEIGHT * IN_SIZEX) + gy * (MIDDLE * IN_WG));
}
)cltag",

// src/cl/fftmiddleout.cl
R"cltag(
// Copyright (C) Mihai Preda and George Woltman

#include "base.cl"
#include "math.cl"
#include "fft-middle.cl"
#include "middle.cl"

KERNEL(OUT_WG) fftMiddleOut(P(T2) out, P(T2) in, Trig trig, BigTab TRIG_BHW) {
  T2 u[MIDDLE];

  u32 SIZEY = OUT_WG / OUT_SIZEX;

  u32 N = SMALL_HEIGHT / OUT_SIZEX;

  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % OUT_SIZEX;
  u32 my = me / OUT_SIZEX;

  // Kernels read OUT_SIZEX consecutive T2.
  // Each WG-thread kernel processes OUT_SIZEX columns from a needed SMALL_HEIGHT columns
  // Each WG-thread kernel processes SIZEY rows out of a needed WIDTH rows

  u32 startx = gx * OUT_SIZEX;  // Each input column increases FFT element by one
  u32 starty = gy * SIZEY;  // Each input row increases FFT element by BIG_HEIGHT
  in += starty * BIG_HEIGHT + startx;

  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = in[i * SMALL_HEIGHT + my * BIG_HEIGHT + mx]; }

  middleMul(u, startx + mx, trig, TRIG_BHW);

  fft_MIDDLE(u);

  // FFT results come out multiplied by the FFT length (NWORDS).  Also, for performance reasons
  // weights and invweights are doubled meaning we need to divide by another 2^2 and 2^2.
  // Finally, roundoff errors are sometimes improved if we use the next lower double precision
  // number.  This may be due to roundoff errors introduced by applying inexact TWO_TO_N_8TH weights.
  double factor = 1.0 / (4 * 4 * NWORDS);

  middleMul2(u, starty + my, startx + mx, factor, trig, TRIG_BHW);
  local T lds[OUT_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];

  middleShuffle(lds, u, OUT_WG, OUT_SIZEX);

  out += MIDDLE * WIDTH * OUT_SIZEX * gx + MIDDLE * OUT_WG * gy;
  out += me;

  for (i32 i = 0; i < MIDDLE; ++i) {
    out[OUT_WG * i] = u[i];
    // out[MIDDLE * OUT_WG * gy + MIDDLE * WIDTH * OUT_SIZEX * gx + OUT_WG * i + me] = u[i];
  }
}
)cltag",

// src/cl/fftp.cl
R"cltag(
// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"
#include "weight.cl"
#include "fftwidth.cl"

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig, BigTab THREAD_WEIGHTS) {
  local T2 lds[WIDTH / 2];

  T2 u[NW];
  u32 g = get_group_id(0);

  u32 step = WIDTH * g;
  in  += step;
  out += step;

  u32 me = get_local_id(0);

  T base = optionalHalve(fancyMul(THREAD_WEIGHTS[me].y, THREAD_WEIGHTS[G_W + g].y));

  for (u32 i = 0; i < NW; ++i) {
    T w1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    T w2 = optionalHalve(fancyMul(w1, WEIGHT_STEP));
    u32 p = G_W * i + me;
    u[i] = U2(in[p].x, in[p].y) * U2(w1, w2);
  }

  fft_WIDTH(lds, u, smallTrig);
  
  write(G_W, NW, u, out, 0);
}
)cltag",

// src/cl/fftw.cl
R"cltag(
// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"
#include "fftwidth.cl"
#include "middle.cl"

// Do an fft_WIDTH after a transposeH (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_W) fftW(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[WIDTH / 2];
  
  T2 u[NW];
  u32 g = get_group_id(0);

  readCarryFusedLine(in, u, g);
  fft_WIDTH(lds, u, smallTrig);  
  out += WIDTH * g;
  write(G_W, NW, u, out, 0);
}
)cltag",

// src/cl/fftwidth.cl
R"cltag(
// Copyright (C) Mihai Preda

#include "fftbase.cl"

void fft_NW(T2 *u) {
#if NW == 4
  fft4(u);
#elif NW == 8
  fft8(u);
#else
#error NW
#endif
}

#if 0
T2 swizzle(T2 src) {
  int4 s = as_int4(src);
  if (NW == 4) {
    const int how = 0x1c;
    s.x = __builtin_amdgcn_ds_swizzle(s.x, how);
    s.y = __builtin_amdgcn_ds_swizzle(s.y, how);
    s.z = __builtin_amdgcn_ds_swizzle(s.z, how);
    s.w = __builtin_amdgcn_ds_swizzle(s.w, how);
  } else {
    const int how = 0x18;
    s.x = __builtin_amdgcn_ds_swizzle(s.x, how);
    s.y = __builtin_amdgcn_ds_swizzle(s.y, how);
    s.z = __builtin_amdgcn_ds_swizzle(s.z, how);
    s.w = __builtin_amdgcn_ds_swizzle(s.w, how);
  }
  return as_double2(s);
}
#endif

void fft_WIDTH(local T2 *lds, T2 *u, Trig trig) {

#if WIDTH != 256 && WIDTH != 512 && WIDTH != 1024 && WIDTH != 4096
#error WIDTH must be one of: 256, 512, 1024, 4096
#endif

  UNROLL_WIDTH_CONTROL
  for (u32 s = 1; s < WIDTH / NW; s *= NW) {
    if (s > 1) { bar(); }
    fft_NW(u);
    tabMul(WIDTH / NW, trig, u, NW, s);
    shufl( WIDTH / NW, lds,  u, NW, s);
  }
  fft_NW(u);
}
)cltag",

// src/cl/math.cl
R"cltag(
// Copyright (C) Mihai Preda

#pragma once

#include "base.cl"

T2 U2(T a, T b) { return (T2) (a, b); }

OVERLOAD T fancyMul(T x, const T y) {
  // x * (y + 1) == x * y + x
  return fma(x, y, x);
}

OVERLOAD T2 fancyMul(T2 x, const T2 y) {
  return U2(fancyMul(RE(x), RE(y)), fancyMul(IM(x), IM(y)));
}

// fma(x, y, z); }
OVERLOAD double mad(double x, double y, double z) { return x * y + z; }

// complex fma
OVERLOAD T2 mad(T2 a, T2 b, T2 c) {
  return U2(mad(a.x, b.x, mad(a.y, -b.y, c.x)), mad(a.x, b.y, mad(a.y, b.x, c.y)));
}

// complex square
T2 sq(T2 a) { return U2(mad(a.x, a.x, - a.y * a.y), 2 * a.x * a.y); }

// a^2 + c
T2 sqa(T2 a, T2 c) { return U2(mad(a.x, a.x, mad(a.y, -a.y, c.x)), mad(2 * a.x, a.y, c.y)); }

// complex mul
OVERLOAD T2 mul(T2 a, T2 b) { return U2(mad(RE(a), RE(b), -IM(a)*IM(b)), mad(RE(a), IM(b), IM(a)*RE(b))); }

// Complex mul a * (b + 1)
// Useful for mul with twiddles of small angles, where the real part is stored with the -1 trick for increased precision
T2 fancyMulTrig(T2 a, T2 b) {
  return U2(
      #if 0
        fma(a.x, b.x, fma(a.y, -b.y, a.x)),
        fma(a.y, b.x, fma(a.x, b.y, a.y))
      #else
        fma(a.y, -b.y, fma(a.x, b.x, a.x)),
        fma(a.x,  b.y, fma(a.y, b.x, a.y))
      #endif
        );
}

T2 fancyMulTrigConj(T2 a, T2 b) {
  return U2(
        fma(a.y, b.y, fma(a.x, b.x, a.x)),
        fma(a.x, -b.y, fma(a.y, b.x, a.y))
        );
}

// Returns complex a * (b + 1) + c
T2 fancyMadTrig(T2 a, T2 b, T2 c) {
  return U2(
        fma(a.y, -b.y, fma(a.x, b.x, a.x + c.x)),
        fma(a.x, b.y, fma(a.y, b.x, a.y + c.y))
        );
}

// Returns complex (a + 1) * (b + 1) - 1 == a * (b + 1) + b
T2 fancyMulUpdate(T2 a, T2 b) { return fancyMadTrig(a, b, b); }

T2 fancySqUpdate(T2 a) {
  return 2 * U2(-a.y * a.y, fma(a.x, a.y, a.y));
  /*
      U2(
        // fma(a.y, -a.y, fma(a.x, a.x, 2 * a.x)),
        fma(a.x, a.x, fma(a.y, -a.y, 2 * a.x)),
        2 * fma(a.x, a.y, a.y)
        );
  */
}

T2 mul_t4(T2 a)  { return U2(IM(a), -RE(a)); } // mul(a, U2( 0, -1)); }

T2 mul_t8(T2 a)  { // mul(a, U2( 1, -1)) * (T)(M_SQRT1_2); }
  return U2(a.y + a.x, a.y - a.x) * M_SQRT1_2;
}

T2 mul_3t8(T2 a) { // mul(a, U2(-1, -1)) * (T)(M_SQRT1_2); }
  return U2(a.y - a.x, -a.y -a.x) * M_SQRT1_2;
}

T2 swap(T2 a)      { return U2(IM(a), RE(a)); }
T2 conjugate(T2 a) { return U2(RE(a), -IM(a)); }

T2 addsub(T2 a) { return U2(RE(a) + IM(a), RE(a) - IM(a)); }

// Same as X2(a, b), b = mul_t4(b)
#define X2_mul_t4(a, b) { T2 t = a; a = t + b; t.x = RE(b) - t.x; RE(b) = t.y - IM(b); IM(b) = t.x; }

#define X2(a, b) { T2 t = a; a = t + b; b = t - b; }

// Same as X2(a, conjugate(b))
#define X2conjb(a, b) { T2 t = a; RE(a) = RE(a) + RE(b); IM(a) = IM(a) - IM(b); RE(b) = t.x - RE(b); IM(b) = t.y + IM(b); }

// Same as X2(a, b), a = conjugate(a)
#define X2conja(a, b) { T2 t = a; RE(a) = RE(a) + RE(b); IM(a) = -IM(a) - IM(b); b = t - b; }

#define SWAP(a, b) { T2 t = a; a = b; b = t; }

T2 fmaT2(T a, T2 b, T2 c) { return a * b + c; }

// Partial complex multiplies:  the mul by sin is delayed so that it can be later propagated to an FMA instruction
// complex mul by cos-i*sin given cos/sin, sin
T2 partial_cmul(T2 a, T c_over_s) { return U2(mad(RE(a), c_over_s, IM(a)), mad(IM(a), c_over_s, -RE(a))); }
// complex mul by cos+i*sin given cos/sin, sin
T2 partial_cmul_conjugate(T2 a, T c_over_s) { return U2(mad(RE(a), c_over_s, -IM(a)), mad(IM(a), c_over_s, RE(a))); }

// a = c + sin * d; b = c - sin * d;
#define fma_addsub(a, b, sin, c, d) { d = sin * d; T2 t = c + d; b = c - d; a = t; }

// a * conjugate(b)
// saves one negation
T2 mul_by_conjugate(T2 a, T2 b) { return U2(RE(a) * RE(b) + IM(a) * IM(b), IM(a) * RE(b) - RE(a) * IM(b)); }
)cltag",

// src/cl/middle.cl
R"cltag(
// Copyright (C) Mihai Preda and George Woltman

/* Tunable paramaters:

IN_WG, OUT_WG: default 256; may try 64, 128, 1024
IN_SIZEX, OUT_SIZEX: default 32 (on AMD), may try 4, 8, 16
*/

#if !IN_WG
#define IN_WG 256
#endif

#if !OUT_WG
#define OUT_WG 256
#endif

#if !IN_SIZEX
#if AMDGPU
#define IN_SIZEX 32
#else // !AMDGPU
#if G_W >= 64
#define IN_SIZEX 4
#else
#define IN_SIZEX 32
#endif
#endif
#endif

#if !OUT_SIZEX

#if AMDGPU
// We realized that these (OUT_WG, OUT_SIZEX) combinations work well: (256, 32) and (64, 8)
// so default OUT_SIZEX relative to OUT_WG
#define OUT_SIZEX (OUT_WG / 8)
#else

#if G_W >= 64
#define OUT_SIZEX 4
#else
#define OUT_SIZEX 32
#endif

#endif
#endif

// Read a line for tailFused or fftHin
// This reads partially transposed datat as written by fftMiddleIn
void readTailFusedLine(CP(T2) in, T2 *u, u32 line) {
  // We go to some length here to avoid dividing by MIDDLE in address calculations.
  // The transPos converted logical line number into physical memory line numbers
  // using this formula:  memline = line / WIDTH + line % WIDTH * MIDDLE.
  // We can compute the 0..9 component of address calculations as line / WIDTH,
  // and the 0,10,20,30,..310 component as (line % WIDTH) % 32 = (line % 32),
  // and the multiple of 320 component as (line % WIDTH) / 32

  u32 me = get_local_id(0);
  u32 SIZEY = IN_WG / IN_SIZEX;

  in += line / WIDTH * IN_WG;
  in += line % IN_SIZEX * SIZEY;
  in += line % WIDTH / IN_SIZEX * (SMALL_HEIGHT / SIZEY) * MIDDLE * IN_WG;

  in += me / SIZEY * MIDDLE * IN_WG + me % SIZEY;
  for (i32 i = 0; i < NH; ++i) { u[i] = in[i * G_H / SIZEY * MIDDLE * IN_WG]; }
}


// Read a line for carryFused or FFTW
void readCarryFusedLine(CP(T2) in, T2 *u, u32 line) {
  u32 me = get_local_id(0);
  u32 WG = OUT_WG; // * OUT_SPACING;
  u32 SIZEY = WG / OUT_SIZEX;

  in += line % OUT_SIZEX * SIZEY
        + line % SMALL_HEIGHT / OUT_SIZEX * WIDTH / SIZEY * MIDDLE * WG
        + line / SMALL_HEIGHT * WG;

  in += me / SIZEY * MIDDLE * WG + me % SIZEY;

  for (i32 i = 0; i < NW; ++i) { u[i] = in[i * G_W / SIZEY * MIDDLE * WG]; }
}
)cltag",

// src/cl/tailmul.cl
R"cltag(
// Copyright (C) Mihai Preda and George Woltman

#include "base.cl"
#include "tailutil.cl"
#include "trig.cl"
#include "fftheight.cl"

// Why does this alternate implementation work?  Let t' be the conjugate of t and note that t*t' = 1.
// Now consider these lines from the original implementation (comments appear alongside):
//      b = mul_by_conjugate(b, t);
//      X2(a, b);					a + bt', a - bt'
//      d = mul_by_conjugate(d, t);
//      X2(c, d);					c + dt', c - dt'
//      a = mul(a, c);					(a+bt')(c+dt') = ac + bct' + adt' + bdt'^2
//      b = mul(b, d);					(a-bt')(c-dt') = ac - bct' - adt' + bdt'^2
//      X2(a, b);					2ac + 2bdt'^2,  2bct' + 2adt'
//      b = mul(b, t);					                2bc + 2ad

void onePairMul(T2* pa, T2* pb, T2* pc, T2* pd, T2 conjugate_t_squared) {
  T2 a = *pa, b = *pb, c = *pc, d = *pd;

  X2conjb(a, b);
  X2conjb(c, d);

  T2 tmp = a;

  a = mad(a, c, mul(mul(b, d), conjugate_t_squared));
  b = mad(b, c, mul(tmp, d));

  X2conja(a, b);

  *pa = a;
  *pb = b;
  *pc = c;
  *pd = d;
}

void pairMul(u32 N, T2 *u, T2 *v, T2 *p, T2 *q, T2 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = conjugate(2 * foo2(u[i], p[i]));
      v[i] = 4 * mul(conjugate(v[i]), conjugate(q[i]));
    } else {
      onePairMul(&u[i], &v[i], &p[i], &q[i], -base_squared);
    }

    if (N == NH) {
      onePairMul(&u[i+NH/2], &v[i+NH/2], &p[i+NH/2], &q[i+NH/2], base_squared);
    }

    // T2 new_base_squared = mul(base_squared, U2(0, -1));
    T2 new_base_squared = U2(IM(base_squared), -RE(base_squared));
    onePairMul(&u[i+NH/4], &v[i+NH/4], &p[i+NH/4], &q[i+NH/4], -new_base_squared);

    if (N == NH) {
      onePairMul(&u[i+3*NH/4], &v[i+3*NH/4], &p[i+3*NH/4], &q[i+3*NH/4], new_base_squared);
    }
  }
}

KERNEL(G_H) tailMul(P(T2) out, CP(T2) in, CP(T2) a, Trig smallTrig, BigTab tailTrig) {
  local T2 lds[SMALL_HEIGHT / 2];

  T2 u[NH], v[NH];
  T2 p[NH], q[NH];

  u32 W = SMALL_HEIGHT;
  u32 H = ND / W;

  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);
    
#if MUL_LOW
  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  read(G_H, NH, p, a, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, a, memline2 * SMALL_HEIGHT);
  fft_HEIGHT(lds, u, smallTrig);
  bar();
  fft_HEIGHT(lds, v, smallTrig);
#else
  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  readTailFusedLine(a, p, line1);
  readTailFusedLine(a, q, line2);
  fft_HEIGHT(lds, u, smallTrig);
  bar();
  fft_HEIGHT(lds, v, smallTrig);
  bar();
  fft_HEIGHT(lds, p, smallTrig);
  bar();
  fft_HEIGHT(lds, q, smallTrig);
#endif

  u32 me = get_local_id(0);

  if (line1 == 0) {

#if TAIL_TAB
    T2 trig1 = tailTrig[me];
#else
    T2 trig1 = slowTrig_N(me * H, ND / NH, NULL);     // slowTrig_2SH(2 * me, SMALL_HEIGHT / 2, TRIG_2SH)
#endif

    T2 trig2 = fancyMulTrig(trig1, tailTrig[G_H]);

    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, trig1, true);
    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);

    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, trig2, false);
    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
  } else {    
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);

#if TAIL_TAB
    T2 trig = fancyMulTrig(tailTrig[me], tailTrig[G_H + line1]);
#else
    T2 trig = slowTrig_N(line1 + me * H, ND / NH, NULL);
#endif

    pairMul(NH, u, v, p, q, trig, false);
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);
  }

  bar();
  fft_HEIGHT(lds, v, smallTrig);
  write(G_H, NH, v, out, memline2 * SMALL_HEIGHT);

  bar();
  fft_HEIGHT(lds, u, smallTrig);
  write(G_H, NH, u, out, memline1 * SMALL_HEIGHT);
}
)cltag",

// src/cl/tailsquare.cl
R"cltag(
// Copyright (C) Mihai Preda and George Woltman

#include "tailutil.cl"
#include "trig.cl"
#include "fftheight.cl"

// Why does this alternate implementation work?  Let t' be the conjugate of t and note that t*t' = 1.
// Now consider these lines from the original implementation (comments appear alongside):
//      b = mul_by_conjugate(b, t); 			bt'
//      X2(a, b);					a + bt', a - bt'
//      a = sq(a);					a^2 + 2abt' + (bt')^2
//      b = sq(b);					a^2 - 2abt' + (bt')^2
//      X2(a, b);					2a^2 + 2(bt')^2, 4abt'
//      b = mul(b, t);					                 4ab

void onePairSq(T2* pa, T2* pb, T2 conjugate_t_squared) {
  T2 a = *pa;
  T2 b = *pb;

  X2conjb(a, b);

  T2 tmp = a;
  a = sqa(a, mul(sq(b), conjugate_t_squared));
  b = 2 * mul(tmp, b);

  X2conja(a, b);

  *pa = a;
  *pb = b;
}

void pairSq(u32 N, T2 *u, T2 *v, T2 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = 2 * foo(conjugate(u[i]));
      v[i] = 4 * sq(conjugate(v[i]));
    } else {
      onePairSq(&u[i], &v[i], -base_squared);
    }

    if (N == NH) {
      onePairSq(&u[i+NH/2], &v[i+NH/2], base_squared);
    }

    // T2 new_base_squared = mul(base_squared, U2(0, -1));
    T2 new_base_squared = U2(base_squared.y, -base_squared.x);
    onePairSq(&u[i+NH/4], &v[i+NH/4], -new_base_squared);

    if (N == NH) {
      onePairSq(&u[i+3*NH/4], &v[i+3*NH/4], new_base_squared);
    }
  }
}

KERNEL(G_H) tailSquare(P(T2) out, CP(T2) in, Trig smallTrig, BigTab tailTrig) {
  local T2 lds[SMALL_HEIGHT / 2];

  T2 u[NH], v[NH];

  // u32 W = SMALL_HEIGHT;
  u32 H = ND / SMALL_HEIGHT;

  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);

  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  fft_HEIGHT(lds, u, smallTrig);
  bar();
  fft_HEIGHT(lds, v, smallTrig);

  u32 me = get_local_id(0);

  if (line1 == 0) {
#if 0 && !TAIL_TAB
    T2 trig1 = slowTrig_N(me * H, ND / NH, NULL);     // slowTrig_2SH(2 * me, SMALL_HEIGHT / 2, TRIG_2SH)
    T2 trig2 = slowTrig_N(H/2 + me * H, ND/NH, NULL); // slowTrig_2SH(1 + 2 * me, SMALL_HEIGHT / 2, TRIG_2SH)
#else
    T2 trigMe   = tailTrig[me];
    T2 trigLine = tailTrig[G_H + line1];
    T2 trig1 = trigMe;
    T2 trig2 = fancyMulTrig(trigMe, trigLine);
#endif

    // Line 0 is special: it pairs with itself, offseted by 1.
    reverse(G_H, lds, u + NH/2, true);
    pairSq(NH/2, u,   u + NH/2, trig1, true);
    reverse(G_H, lds, u + NH/2, true);

    // Line H/2 also pairs with itself (but without offset).
    reverse(G_H, lds, v + NH/2, false);
    pairSq(NH/2, v,   v + NH/2, trig2, false);
    reverse(G_H, lds, v + NH/2, false);
  } else {    
    reverseLine(G_H, lds, v);

#if !TAIL_TAB
    T2 trig = slowTrig_N(line1 + me * H, ND / NH, NULL);
#else
    T2 trigMe   = tailTrig[me];
    T2 trigLine = tailTrig[G_H + line1];
    T2 trig = fancyMulTrig(trigMe, trigLine);
    // tailTrig[line1 * G_H + me]
#endif
    pairSq(NH, u, v, trig, false);
    reverseLine(G_H, lds, v);
  }

  bar();
  fft_HEIGHT(lds, v, smallTrig);
  bar();
  fft_HEIGHT(lds, u, smallTrig);
  write(G_H, NH, v, out, memline2 * SMALL_HEIGHT);
  write(G_H, NH, u, out, memline1 * SMALL_HEIGHT);
}
)cltag",

// src/cl/tailutil.cl
R"cltag(
// Copyright (C) Mihai Preda

#include "math.cl"

void reverse(u32 WG, local T2 *lds, T2 *u, bool bump) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me + bump;

  bar();

#if NH == 8
  lds[revMe + 0 * WG] = u[3];
  lds[revMe + 1 * WG] = u[2];
  lds[revMe + 2 * WG] = u[1];
  lds[bump ? ((revMe + 3 * WG) % (4 * WG)) : (revMe + 3 * WG)] = u[0];
#elif NH == 4
  lds[revMe + 0 * WG] = u[1];
  lds[bump ? ((revMe + WG) % (2 * WG)) : (revMe + WG)] = u[0];
#else
#error
#endif

  bar();
  for (i32 i = 0; i < NH/2; ++i) { u[i] = lds[i * WG + me]; }
}

void reverseLine(u32 WG, local T2 *lds, T2 *u) {
  u32 me = get_local_id(0);
  u32 revMe = WG - 1 - me;

  for (i32 b = 0; b < 2; ++b) {
    bar();
    for (i32 i = 0; i < NH; ++i) { ((local T*)lds)[i * WG + revMe] = ((T *) (u + ((NH - 1) - i)))[b]; }
    bar();
    for (i32 i = 0; i < NH; ++i) { ((T *) (u + i))[b] = ((local T*)lds)[i * WG + me]; }
  }
}

// computes 2*(a.x*b.x+a.y*b.y) + i*2*(a.x*b.y+a.y*b.x)
// which happens to be the cyclical convolution (a.x, a.y)x(b.x, b.y) * 2
T2 foo2(T2 a, T2 b) {
  a = addsub(a);
  b = addsub(b);
  return addsub(U2(RE(a) * RE(b), IM(a) * IM(b)));
}

// computes 2*[x^2+y^2 + i*(2*x*y)]. i.e. 2 * cyclical autoconvolution of (x, y)
T2 foo(T2 a) { return foo2(a, a); }
)cltag",

// src/cl/transpose.cl
R"cltag(
// Copyright (C) Mihai Preda

#include "base.cl"

void transposeWords(u32 W, u32 H, local Word2 *lds, const Word2 *in, Word2 *out) {
  u32 GPW = W / 64, GPH = H / 64;

  u32 g = get_group_id(0);
  u32 gy = g % GPH;
  u32 gx = g / GPH;
  gx = (gy + gx) % GPW;

  in   += 64 * W * gy + 64 * gx;
  out  += 64 * gy + 64 * H * gx;
  u32 me = get_local_id(0);
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    lds[i * 64 + me] = in[i * W + me];
  }
  bar();
  #pragma unroll 1
  for (i32 i = 0; i < 64; ++i) {
    out[i * H + me] = lds[me * 64 + i];
  }
}

// from transposed to sequential.
KERNEL(64) transposeOut(P(Word2) out, CP(Word2) in) {
  local Word2 lds[4096];
  transposeWords(WIDTH, BIG_HEIGHT, lds, in, out);
}

// from sequential to transposed.
KERNEL(64) transposeIn(P(Word2) out, CP(Word2) in) {
  local Word2 lds[4096];
  transposeWords(BIG_HEIGHT, WIDTH, lds, in, out);
}
)cltag",

// src/cl/trig.cl
R"cltag(
// Copyright (C) George Woltman and Mihai Preda

#pragma once

#if ULTRA_TRIG

// These are ultra accurate routines.  We modified Ernst's qfcheb program and selected a multiplier such that
// a) k * multipler / n can be represented exactly as a double, and
// b) x * x can be represented exactly as a double, and
// c) the difference between S0 and C0 represented as a double vs infinite precision is minimized.
// Note that condition (a) requires different multipliers for different MIDDLE values.

#if MIDDLE <= 4 || MIDDLE == 6 || MIDDLE == 8 || MIDDLE == 12

#define SIN_COEFS {0.013255665205020225,-3.8819803226819742e-07,3.4105654433606424e-12,-1.4268560139781677e-17,3.4821751757020666e-23,-5.5620764489252689e-29,6.2011635226098908e-35, 237}
#define COS_COEFS {-8.7856330013791936e-05,1.2864557872487131e-09,-7.5348856128299892e-15,2.3642407019488875e-20,-4.6158547847666762e-26,6.1440808274170587e-32,-5.8714657758002626e-38, 237}

#elif MIDDLE == 11

#define SIN_COEFS {0.0058285577988678909,-3.3001377814174114e-08,5.6056282285321817e-14,-4.5341639101320423e-20,2.1393746357239815e-26,-6.6068179928427645e-33,1.4241237670800455e-39, 49*11}

// {0.005492294848933205,-2.761278944626038e-08,4.1647407612374865e-14,-2.9912063263788177e-20,1.2532031863362935e-26,-3.4364949357849832e-33,6.5816772637974481e-40, 143 * 4};

#define COS_COEFS {-1.6986043007371857e-05,4.8087609508047477e-11,-5.4454546881584527e-17,3.3034545522108849e-23,-1.2469468591680302e-29,3.2090160573145604e-36,-5.9289892824449386e-43, 49*11}

// {-1.5082651353809108e-05,3.7914395310092582e-11,-3.8123307049954028e-17,2.0535733847563737e-23,-6.8829596395036851e-30,1.5727926864212422e-36,-2.5737325744028274e-43, 143 * 4};

// This should be the best choice for MIDDLE=11.  For reasons I cannot explain, the Sun coefficients beat this
// code.  We know this code works as it gives great results for MIDDLE=5 and MIDDLEE=10.
#elif MIDDLE == 5 || MIDDLE == 10 || MIDDLE == 11

#define SIN_COEFS {0.0033599921428767842,-6.3221316482145663e-09,3.5687001824123009e-15,-9.5926212207432193e-22,1.5041156546369205e-28,-1.5436222869257443e-35,1.1057341951605355e-42, 85 * 11}
#define COS_COEFS {-5.6447736000968621e-06,5.3105781660583875e-12,-1.9984674288638241e-18,4.0288914910974918e-25,-5.0538167304629085e-32,4.3221291704923216e-39,-2.6537550015407366e-46, 85 * 11}

#elif MIDDLE == 7 || MIDDLE == 14

#define SIN_COEFS {0.0030120734933746819,-4.5545496673734544e-09,2.066077343547647e-15,-4.4630156850662332e-22,5.6237622654854882e-29,-4.6381134477150518e-36,2.6699656391050201e-43, 149 * 7}
#define COS_COEFS {-4.5362933647451799e-06,3.4296595818385068e-12,-1.0371961336265129e-18,1.6803664055570525e-25,-1.6939185081650983e-32,1.1641940392856976e-39,-5.7443786570712859e-47, 149 * 7}

#elif MIDDLE == 9

#define SIN_COEFS {0.0032024389944850084,-5.4738305054252556e-09,2.8068750524532041e-15,-6.8538645985346134e-22,9.7625812823195236e-29,-9.1014309535685973e-36,5.9224934064240749e-43, 109*9}
#define COS_COEFS {-5.1278077566990758e-06,4.3824020649438457e-12,-1.4981410201032161e-18,2.7436354064447543e-25,-3.1264070669243379e-32,2.4288963593034543e-39,-1.3547441195887408e-46,109*9}

#elif MIDDLE == 13

#define SIN_COEFS {0.0034036756810290284,-6.5719350793639721e-09,3.8067960700283384e-15,-1.0500419865908618e-21,1.6895475541832181e-28,-1.779303563885441e-35,1.3079151443222568e-42, 71*13}
#define COS_COEFS {-5.7925040708142102e-06,5.5921839017331711e-12,-2.1595165343644285e-18,4.4675029669254463e-25,-5.7506718639750315e-32,5.0468065746580596e-39,-3.1797982320621979e-46, 71*13}

#elif MIDDLE == 15

#define SIN_COEFS {0.0032221463113741469,-5.5755089924509359e-09,2.8943099587177684e-15,-7.1546148933480147e-22,1.0316780436916074e-28,-9.7368389615915834e-36,6.4141878934890016e-43, 65*15}
#define COS_COEFS {-5.1911134259510105e-06,4.4912764335147829e-12,-1.5543150262419615e-18,2.8816519982635366e-25,-3.3242175693980929e-32,2.6144580878684214e-39,-1.4762460824550342e-46, 65*15}

#endif

double ksinpi(u32 k, u32 n) {
  const double S[] = SIN_COEFS;

  double x = S[7] / n * k;
  double z = x * x;
  double r = fma(fma(fma(fma(fma(S[6], z, S[5]), z, S[4]), z, S[3]), z, S[2]), z, S[1]) * (z * x);
  return fma(x, S[0], r);
}

#else

// Copyright notice of original k_cos, k_sin from which our ksin/kcos evolved:
/* ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

// Coefficients from http://www.netlib.org/fdlibm/k_cos.c
#define COS_COEFS {-0.5,0.041666666666666602,-0.001388888888887411,2.4801587289476729e-05,-2.7557314351390663e-07,2.0875723212981748e-09,-1.1359647557788195e-11, M_PI}

// Experimental: const double C[] = {-0.5,0.041666666666665589,-0.0013888888888779014,2.4801587246942509e-05,-2.7557304501248813e-07,2.0874583610048953e-09,-1.1307548621486489e-11, M_PI};

double ksinpi(u32 k, u32 n) {
  // const double S[] = {-0.16666666666666455,0.0083333333332988729,-0.00019841269816529426,2.7557310051600518e-06,-2.5050279451232251e-08,1.5872611854244144e-10};

  // Coefficients from http://www.netlib.org/fdlibm/k_sin.c
  const double S[] = {1, -0.16666666666666666,0.0083333333333309497,-0.00019841269836761127,2.7557316103728802e-06,-2.5051132068021698e-08,1.5918144304485914e-10, M_PI};

  double x = S[7] / n * k;
  double z = x * x;
  // Special-case based on S[0]==1:
  return fma(fma(fma(fma(fma(fma(S[6], z, S[5]), z, S[4]), z, S[3]), z, S[2]), z, S[1]), z * x, x);
}

#endif

double kcospi(u32 k, u32 n) {
  const double C[] = COS_COEFS;
  double x = C[7] / n * k;
  double z = x * x;
  return fma(fma(fma(fma(fma(fma(fma(C[6], z, C[5]), z, C[4]), z, C[3]), z, C[2]), z, C[1]), z, C[0]), z, 1);
}

// N represents a full circle, so N/2 is pi radians and N/8 is pi/4 radians.
double2 reducedCosSin(u32 k, u32 N) {
  assert(k <= N/8);
  return U2(kcospi(k, N/2), -ksinpi(k, N/2));
}

#if !defined(TRIG_TAB)
#define TRIG_TAB 0
#endif

// Returns e^(-i * tau * k / n), (tau == 2*pi represents a full circle). So k/n is the ratio of a full circle.
// Inverse trigonometric direction is chosen as an FFT convention.
double2 slowTrig_N(u32 k, u32 kBound, BigTab TRIG_BHW)   {
  u32 n = ND;
  assert(n % 8 == 0);
  assert(k < kBound);       // kBound actually bounds k
  assert(kBound <= 2 * n);  // angle <= 2 tau

  if (kBound > n && k >= n) { k -= n; }
  assert(k < n);

  bool negate = kBound > n/2 && k >= n/2;
  if (negate) { k -= n/2; }

  bool negateCos = kBound > n / 4 && k >= n / 4;
  if (negateCos) { k = n/2 - k; }

  bool flip = kBound > n / 8 + 1 && k > n / 8;
  if (flip) { k = n / 4 - k; }

  assert(k <= n / 8);

  double2 r;

  if (TRIG_TAB & (TRIG_BHW != NULL)) { // bitwise because annoying warning -Wconstant-logical-operand
    u32 a = (k + WIDTH/2) / WIDTH;
    i32 b = k - a * WIDTH;

    double2 cs1 = TRIG_BHW[a];
    if (b == 0) {
      r = cs1;
    } else {
      double2 cs2 = TRIG_BHW[BIG_HEIGHT/8 + 1 + abs(b)];
      r = fancyMulTrig(cs1, b < 0 ? conjugate(cs2) : cs2);
    }
  } else {
    r = reducedCosSin(k, n);
  }

  if (flip) { r = -swap(r); }
  if (negateCos) { r.x = -r.x; }
  if (negate) { r = -r; }

  return r;
}
)cltag",

// src/cl/weight.cl
R"cltag(
// Copyright (C) Mihai Preda and George Woltman

#define STEP (NWORDS - (EXP % NWORDS))
// bool isBigWord(u32 extra) { return extra < NWORDS - STEP; }

T fweightStep(u32 i) {
  const T TWO_TO_NTH[8] = {
    // 2^(k/8) -1 for k in [0..8)
    0,
    0.090507732665257662,
    0.18920711500272105,
    0.29683955465100964,
    0.41421356237309503,
    0.54221082540794086,
    0.68179283050742912,
    0.83400808640934243,
  };
  return TWO_TO_NTH[i * STEP % NW * (8 / NW)];
}

T iweightStep(u32 i) {
  const T TWO_TO_MINUS_NTH[8] = {
    // 2^-(k/8) - 1 for k in [0..8)
    0,
    -0.082995956795328771,
    -0.15910358474628547,
    -0.2288945872960296,
    -0.29289321881345248,
    -0.35158022267449518,
    -0.40539644249863949,
    -0.45474613366737116,
  };
  return TWO_TO_MINUS_NTH[i * STEP % NW * (8 / NW)];
}

u32 bfi(u32 u, u32 mask, u32 bits) {
#if HAS_ASM
  u32 out;
  __asm("v_bfi_b32 %0, %1, %2, %3" : "=v"(out) : "v"(mask), "v"(u), "v"(bits));
  return out;
#else
  // return (u & mask) | (bits & ~mask);
  return (u & mask) | bits;
#endif
}

T optionalDouble(T iw) {
  // In a straightforward implementation, inverse weights are between 0.5 and 1.0.  We use inverse weights between 1.0 and 2.0
  // because it allows us to implement this routine with a single OR instruction on the exponent.   The original implementation
  // where this routine took as input values from 0.25 to 1.0 required both an AND and an OR instruction on the exponent.
  // return iw <= 1.0 ? iw * 2 : iw;
  assert(iw > 0.5 && iw < 2);
  uint2 u = as_uint2(iw);

  u.y |= 0x00100000;
  // u.y = bfi(u.y, 0xffefffff, 0x00100000);

  return as_double(u);
}

T optionalHalve(T w) {    // return w >= 4 ? w / 2 : w;
  // In a straightforward implementation, weights are between 1.0 and 2.0.  We use weights between 2.0 and 4.0 because
  // it allows us to implement this routine with a single AND instruction on the exponent.   The original implementation
  // where this routine took as input values from 1.0 to 4.0 required both an AND and an OR instruction on the exponent.
  assert(w >= 2 && w < 8);
  uint2 u = as_uint2(w);
  // u.y &= 0xFFEFFFFF;
  u.y = bfi(u.y, 0xffefffff, 0);
  return as_double(u);
}
)cltag",

};
static const std::vector<const char*> CL_FILE_NAMES{"base.cl","carry.cl","carryb.cl","carryfused.cl","carryinc.cl","carryutil.cl","etc.cl","fft-middle.cl","fft10.cl","fft11.cl","fft12.cl","fft13.cl","fft14.cl","fft15.cl","fft3.cl","fft4.cl","fft5.cl","fft6.cl","fft7.cl","fft8.cl","fft9.cl","fftbase.cl","fftheight.cl","ffthin.cl","fftmiddlein.cl","fftmiddleout.cl","fftp.cl","fftw.cl","fftwidth.cl","math.cl","middle.cl","tailmul.cl","tailsquare.cl","tailutil.cl","transpose.cl","trig.cl","weight.cl",};
const std::vector<const char*>& getClFileNames() { return CL_FILE_NAMES; }
const std::vector<const char*>& getClFiles() { return CL_FILES; }

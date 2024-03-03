// Copyright (C) Mihai Preda and George Woltman.

/* List of user-serviceable -use flags and their effects : see also help (-h)

OUT_WG,OUT_SIZEX,OUT_SPACING <AMD default is 256,32,4> <nVidia default is 256,4,1 but needs testing>
IN_WG,IN_SIZEX,IN_SPACING <AMD default is 256,32,1>  <nVidia default is 256,4,1 but needs testing>

UNROLL_WIDTH <nVidia default>
NO_UNROLL_WIDTH <AMD default>

OLD_FFT5
NEW_FFT5 <default>
NEWEST_FFT5

NEW_FFT9 <default>
OLD_FFT9
*/

/* List of *derived* binary macros. These are normally not defined through -use flags, but derived.
AMDGPU  : set on AMD GPUs
HAS_ASM : set if we believe __asm() can be used
 */

/* List of code-specific macros. These are set by the C++ host code or derived
EXP        the exponent
WIDTH
SMALL_HEIGHT
MIDDLE

-- Derived from above:
BIG_HEIGHT == SMALL_HEIGHT * MIDDLE
ND         number of dwords == WIDTH * MIDDLE * SMALL_HEIGHT
NWORDS     number of words  == ND * 2
G_W        "group width" (e.g. 256)
G_H        "group height"
NW         == WIDTH / G_W
NH         == SMALL_HEIGHT / G_H
 */

#if !defined(TRIG_COMPUTE)
#define TRIG_COMPUTE 2
#endif

#define STR(x) XSTR(x)
#define XSTR(x) #x

#define OVERLOAD __attribute__((overloadable))

#pragma OPENCL FP_CONTRACT ON

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
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

#if WIDTH == 1024 || WIDTH == 256
#define NW 4
#else
#define NW 8
#endif

#if SMALL_HEIGHT == 1024 || SMALL_HEIGHT == 256
#define NH 4
#else
#define NH 8
#endif

#define G_W (WIDTH / NW)
#define G_H (SMALL_HEIGHT / NH)

// 5M timings for MiddleOut & carryFused, ROCm 2.10, RadeonVII, sclk4, mem 1200
// OUT_WG=256, OUT_SIZEX=4, OUT_SPACING=1 (old WorkingOut4) : 154 + 252 = 406 (but may be best on nVidia)
// OUT_WG=256, OUT_SIZEX=8, OUT_SPACING=1 (old WorkingOut3): 124 + 260 = 384
// OUT_WG=256, OUT_SIZEX=32, OUT_SPACING=1 (old WorkingOut5): 105 + 281 = 386
// OUT_WG=256, OUT_SIZEX=8, OUT_SPACING=2: 122 + 249 = 371
// OUT_WG=256, OUT_SIZEX=32, OUT_SPACING=4: 108 + 257 = 365  <- best

#if !OUT_WG
#define OUT_WG 256
#endif

#if !OUT_SIZEX
#if AMDGPU
#define OUT_SIZEX 32
#else // AMDGPU
#if G_W >= 64
#define OUT_SIZEX 4
#else
#define OUT_SIZEX 32
#endif
#endif
#endif

#if !OUT_SPACING
#if AMDGPU
#define OUT_SPACING 4
#else
#define OUT_SPACING 1
#endif
#endif

#if !IN_WG
#define IN_WG 256
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
typedef double2 TT;
#define RE(a) (a.x)
#define IM(a) (a.y)

void bar() {
  // barrier(CLK_LOCAL_MEM_FENCE) is correct, but it turns out that on some GPUs, in particular on RadeonVII,
  // barrier(0) works as well and is faster. So allow selecting the faster path when it works with
  // -use FAST_BARRIER
#if FAST_BARRIER
  barrier(0);
#else
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
}

TT U2(T a, T b) { return (TT) (a, b); }

OVERLOAD double sum(double a, double b) { return a + b; }

OVERLOAD double mad1(double x, double y, double z) { return x * y + z; }
  // fma(x, y, z); }

OVERLOAD double mul(double x, double y) { return x * y; }

T add1_m2(T x, T y) {
   return 2 * sum(x, y);
}

T sub1_m2(T x, T y) {
  return 2 * sum(x, -y);
}

// x * y * 2
T mul1_m2(T x, T y) {
  return 2 * mul(x, y);
}


OVERLOAD T fancyMul(T x, const T y) {
  // x * (y + 1);
  return fma(x, y, x);
}

OVERLOAD TT fancyMul(TT x, const TT y) {
  return U2(fancyMul(RE(x), RE(y)), fancyMul(IM(x), IM(y)));
}

T mad1_m2(T a, T b, T c) {
  return 2 * mad1(a, b, c);
}

T mad1_m4(T a, T b, T c) {
  return 4 * mad1(a, b, c);
}

// complex square
OVERLOAD TT sq(TT a) { return U2(mad1(RE(a), RE(a), - IM(a) * IM(a)), mul1_m2(RE(a), IM(a))); }

// complex mul
OVERLOAD TT mul(TT a, TT b) { return U2(mad1(RE(a), RE(b), - IM(a) * IM(b)), mad1(RE(a), IM(b), IM(a) * RE(b))); }

bool test(u32 bits, u32 pos) { return (bits >> pos) & 1; }

#define STEP (NWORDS - (EXP % NWORDS))
// bool isBigWord(u32 extra) { return extra < NWORDS - STEP; }

u32 bitlen(bool b) { return EXP / NWORDS + b; }


// complex add * 2
TT add_m2(TT a, TT b) { return U2(add1_m2(RE(a), RE(b)), add1_m2(IM(a), IM(b))); }

// complex mul * 2
TT mul_m2(TT a, TT b) { return U2(mad1_m2(RE(a), RE(b), -mul(IM(a), IM(b))), mad1_m2(RE(a), IM(b), mul(IM(a), RE(b)))); }

// complex mul * 4
TT mul_m4(TT a, TT b) { return U2(mad1_m4(RE(a), RE(b), -mul(IM(a), IM(b))), mad1_m4(RE(a), IM(b), mul(IM(a), RE(b)))); }

// complex fma
TT mad_m1(TT a, TT b, TT c) { return U2(mad1(RE(a), RE(b), mad1(IM(a), -IM(b), RE(c))), mad1(RE(a), IM(b), mad1(IM(a), RE(b), IM(c)))); }

// complex fma * 2
TT mad_m2(TT a, TT b, TT c) { return U2(mad1_m2(RE(a), RE(b), mad1(IM(a), -IM(b), RE(c))), mad1_m2(RE(a), IM(b), mad1(IM(a), RE(b), IM(c)))); }

TT mul_t4(TT a)  { return U2(IM(a), -RE(a)); } // mul(a, U2( 0, -1)); }


TT mul_t8(TT a)  { return U2(IM(a) + RE(a), IM(a) - RE(a)) *   M_SQRT1_2; }  // mul(a, U2( 1, -1)) * (T)(M_SQRT1_2); }
TT mul_3t8(TT a) { return U2(RE(a) - IM(a), RE(a) + IM(a)) * - M_SQRT1_2; }  // mul(a, U2(-1, -1)) * (T)(M_SQRT1_2); }

TT swap(TT a)      { return U2(IM(a), RE(a)); }
TT conjugate(TT a) { return U2(RE(a), -IM(a)); }

TT weight(Word2 a, TT w) { return w * U2(RE(a), IM(a)); }

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

#if HAS_ASM
i32  lowBits(i32 u, u32 bits) { i32 tmp; __asm("v_bfe_i32 %0, %1, 0, %2" : "=v" (tmp) : "v" (u), "v" (bits)); return tmp; }
i32 xtract32(i64 x, u32 bits) { i32 tmp; __asm("v_alignbit_b32 %0, %1, %2, %3" : "=v"(tmp) : "v"(as_int2(x).y), "v"(as_int2(x).x), "v"(bits)); return tmp; }
#else
i32  lowBits(i32 u, u32 bits) { return ((u << (32 - bits)) >> (32 - bits)); }
i32 xtract32(i64 x, u32 bits) { return x >> bits; }
#endif

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

// Check for 32-bit carry nearing the limit (default is 0x7C000000)
void CARRY32_CHECK(i32 x) {
#if DEBUG
#ifndef CARRY32_LIMIT
#define CARRY32_LIMIT 0x7C000000
#endif
  if (abs(x) > CARRY32_LIMIT) { printf("Carry32: %X\n", abs(x)); }
#endif
}

// Rounding constant: 3 * 2^51, See https://stackoverflow.com/questions/17035464
#define RNDVAL (3.0 * (1l << 51))

i64 doubleToLong(double x, float* maxROE) {
  // Unfortunatelly (i64) rint() is slow!
  // return rint(x);

  ROUNDOFF_CHECK(x);

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

#if CARRY32
typedef i32 CFcarry;
#else
typedef i64 CFcarry;
#endif

// In the carryMul situation there's an additional multiplication by 3, adding about 1.6bits to carry, so 32bits
// is often not enough.
typedef i64 CFMcarry;

// map abs(carry) to floats, with 2^32 corresponding to 1.0
float OVERLOAD boundCarry(i32 c) { return ldexp(fabs((float) c), -32); }
float OVERLOAD boundCarry(i64 c) { return ldexp(fabs((float) (i32) (c >> 8)), -24); }

typedef TT T2;

#define iCARRY i32
#include "carryutil.cl"
#undef iCARRY

#define iCARRY i64
#include "carryutil.cl"
#undef iCARRY

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

// Propagate carry this many pairs of words.
#define CARRY_LEN 8

T2 addsub(T2 a) { return U2(RE(a) + IM(a), RE(a) - IM(a)); }
T2 addsub_m2(T2 a) { return U2(add1_m2(RE(a), IM(a)), sub1_m2(RE(a), IM(a))); }

// computes 2*(a.x*b.x+a.y*b.y) + i*2*(a.x*b.y+a.y*b.x)
// which happens to be the cyclical convolution (a.x, a.y)x(b.x, b.y) * 2
T2 foo2(T2 a, T2 b) {
  a = addsub(a);
  b = addsub(b);
  return addsub(U2(RE(a) * RE(b), IM(a) * IM(b)));
}

T2 foo2_m2(T2 a, T2 b) {
  a = addsub(a);
  b = addsub(b);
  return addsub_m2(U2(RE(a) * RE(b), IM(a) * IM(b)));
}

// computes 2*[x^2+y^2 + i*(2*x*y)]. i.e. 2 * cyclical autoconvolution of (x, y)
T2 foo(T2 a) { return foo2(a, a); }
T2 foo_m2(T2 a) { return foo2_m2(a, a); }

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
T2 partial_cmul(T2 a, T c_over_s) { return U2(mad1(RE(a), c_over_s, IM(a)), mad1(IM(a), c_over_s, -RE(a))); }
// complex mul by cos+i*sin given cos/sin, sin
T2 partial_cmul_conjugate(T2 a, T c_over_s) { return U2(mad1(RE(a), c_over_s, -IM(a)), mad1(IM(a), c_over_s, RE(a))); }

// a = c + sin * d; b = c - sin * d;
#define fma_addsub(a, b, sin, c, d) { d = sin * d; T2 t = c + d; b = c - d; a = t; }

// a * conjugate(b)
// saves one negation
T2 mul_by_conjugate(T2 a, T2 b) { return U2(RE(a) * RE(b) + IM(a) * IM(b), IM(a) * RE(b) - RE(a) * IM(b)); }

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

void fft2(T2* u) {
  X2(u[0], u[1]);
}

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

// FFT routines to implement the middle step

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

#include "fft5.cl"
#include "fft6.cl"
#include "fft7.cl"
#include "fft9.cl"
#include "fft10.cl"
#include "fft11.cl"
#include "fft12.cl"
#include "fft13.cl"
#include "fft14.cl"
#include "fft15.cl"

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

#if AMDGPU
typedef constant const T2* Trig;
typedef constant const double2* BigTab;
#else
typedef global const T2* Trig;
typedef global const double2* BigTab;
#endif

void tabMul(u32 WG, Trig trig, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  
  for (u32 i = 1; i < n; ++i) {
#if 1
    u[i] = mul(u[i], trig[(me & ~(f-1)) + (i - 1) * WG]);
#else
    u[i] = mul(u[i], trig[WG/f * i + (me / f)]);
#endif
  }
}

void shuflAndMul(u32 WG, local T2 *lds, Trig trig, T2 *u, u32 n, u32 f) {
  tabMul(WG, trig, u, n, f);
  shufl(WG, lds, u, n, f);
}

// 64x4
void fft256w(local T2 *lds, T2 *u, Trig trig) {
  UNROLL_WIDTH_CONTROL
  for (u32 s = 0; s <= 4; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul(64, lds, trig, u, 4, 1u << s);
  }
  fft4(u);
}

void fft256h(local T2 *lds, T2 *u, Trig trig) {
  for (u32 s = 0; s <= 4; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul(64, lds, trig, u, 4, 1u << s);
  }
  fft4(u);
}

// 64x8
void fft512w(local T2 *lds, T2 *u, Trig trig) {
  UNROLL_WIDTH_CONTROL
  for (u32 s = 0; s <= 3; s += 3) {
    if (s) { bar(); }
    fft8(u);
    shuflAndMul(64, lds, trig, u, 8, 1u << s);
  }
  fft8(u);
}

void fft512h(local T2 *lds, T2 *u, Trig trig) {
  for (u32 s = 0; s <= 3; s += 3) {
    if (s) { bar(); }
    fft8(u);
    shuflAndMul(64, lds, trig, u, 8, 1u << s);
  }
  fft8(u);
}

// 256x4
void fft1Kw(local T2 *lds, T2 *u, Trig trig) {
  UNROLL_WIDTH_CONTROL
  for (i32 s = 0; s <= 6; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul(256, lds, trig, u, 4, 1u << s);
  }
  fft4(u);
}

void fft1Kh(local T2 *lds, T2 *u, Trig trig) {
  for (i32 s = 0; s <= 6; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul(256, lds, trig, u, 4, 1u << s);
  }
  fft4(u);
}

// 512x8
void fft4Kw(local T2 *lds, T2 *u, Trig trig) {
  UNROLL_WIDTH_CONTROL
  for (u32 s = 0; s <= 6; s += 3) {
    if (s) { bar(); }
    fft8(u);
    shuflAndMul(512, lds, trig, u, 8, 1u << s);
  }
  fft8(u);
}

void fft4Kh(local T2 *lds, T2 *u, Trig trig) {
  for (u32 s = 0; s <= 6; s += 3) {
    if (s) { bar(); }
    fft8(u);
    shuflAndMul(512, lds, trig, u, 8, 1u << s);
  }
  fft8(u);
}

void read(u32 WG, u32 N, T2 *u, const global T2 *in, u32 base) {
  for (i32 i = 0; i < N; ++i) { u[i] = in[base + i * WG + (u32) get_local_id(0)]; }
}

void write(u32 WG, u32 N, T2 *u, global T2 *out, u32 base) {
  for (i32 i = 0; i < N; ++i) { out[base + i * WG + (u32) get_local_id(0)] = u[i]; }
}

void readDelta(u32 WG, u32 N, T2 *u, const global T2 *a, const global T2 *b, u32 base) {
  for (u32 i = 0; i < N; ++i) {
    u32 pos = base + i * WG + (u32) get_local_id(0); 
    u[i] = a[pos] - b[pos];
  }
}

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

#if TRIG_COMPUTE == 0
global double2 TRIG_N[ND / 8 + 1];
#elif TRIG_COMPUTE == 1
global double2 TRIG_W[WIDTH / 2 + 1];
#endif

double2 tableTrig(u32 k, u32 n, u32 kBound, BigTab trigTable) {
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

  double2 r = trigTable[k];

  if (flip) { r = -swap(r); }
  if (negateCos) { r.x = -r.x; }
  if (negate) { r = -r; }
  return r;
}

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

double2 slowTrig_2SH(u32 k, u32 kBound, BigTab TRIG_2SH) { return tableTrig(k, 2 * SMALL_HEIGHT, kBound, TRIG_2SH); }
double2 slowTrig_BH(u32 k, u32 kBound, BigTab TRIG_BH)  { return tableTrig(k, BIG_HEIGHT, kBound, TRIG_BH); }

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

#if TRIG_COMPUTE >= 2
  double2 r = reducedCosSin(k, n);
#else // TRIG_COMPUTE == 1 or TRIG_COMPUTE == 0
  u32 a = (k + WIDTH/2) / WIDTH;
  i32 b = k - a * WIDTH;
  
  // double2 cs1 = TRIG_BH[a];
  double2 cs1 = TRIG_BHW[a];
  double c1 = cs1.x;
  double s1 = cs1.y;
  
  // double2 cs2 = TRIG_W[abs(b)];
  double2 cs2 = TRIG_BHW[BIG_HEIGHT/8 + 1 + abs(b)];
  double c2 = cs2.x;
  double s2 = (b < 0) ? -cs2.y : cs2.y; 

  // cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
  // sin(a+b) = cos(a)sin(b) + sin(a)cos(b)
  // c2 is stored with "-1" trick to increase accuracy, so we use fma(x,y,x) for x*(y+1)
  double c = fma(-s1, s2, fma(c1, c2, c1));
  double s = fma(c1, s2, fma(s1, c2, s1));
  double2 r = (double2)(c, s);
#endif

  if (flip) { r = -swap(r); }
  if (negateCos) { r.x = -r.x; }
  if (negate) { r = -r; }
  
  return r;
}

#define P(x) global x * restrict
#define CP(x) const P(x)

u32 transPos(u32 k, u32 middle, u32 width) { return k / width + k % width * middle; }

void fft_WIDTH(local T2 *lds, T2 *u, Trig trig) {
#if WIDTH == 256
  fft256w(lds, u, trig);
#elif WIDTH == 512
  fft512w(lds, u, trig);
#elif WIDTH == 1024
  fft1Kw(lds, u, trig);
#elif WIDTH == 4096
  fft4Kw(lds, u, trig);
#else
#error unexpected WIDTH.  
#endif  
}

void fft_HEIGHT(local T2 *lds, T2 *u, Trig trig) {
#if SMALL_HEIGHT == 256
  fft256h(lds, u, trig);
#elif SMALL_HEIGHT == 512
  fft512h(lds, u, trig);
#elif SMALL_HEIGHT == 1024
  fft1Kh(lds, u, trig);
#else
#error unexpected SMALL_HEIGHT.
#endif
}

// Read a line for carryFused or FFTW
void readCarryFusedLine(CP(T2) in, T2 *u, u32 line) {
  u32 me = get_local_id(0);
  u32 WG = OUT_WG * OUT_SPACING;
  u32 SIZEY = WG / OUT_SIZEX;

  in += line % OUT_SIZEX * SIZEY + line % SMALL_HEIGHT / OUT_SIZEX * WIDTH / SIZEY * MIDDLE * WG + line / SMALL_HEIGHT * WG;
  in += me / SIZEY * MIDDLE * WG + me % SIZEY;
  for (i32 i = 0; i < NW; ++i) { u[i] = in[i * G_W / SIZEY * MIDDLE * WG]; }
}

// Read a line for tailFused or fftHin
void readTailFusedLine(CP(T2) in, T2 *u, u32 line) {
  // We go to some length here to avoid dividing by MIDDLE in address calculations.
  // The transPos converted logical line number into physical memory line numbers
  // using this formula:  memline = line / WIDTH + line % WIDTH * MIDDLE.
  // We can compute the 0..9 component of address calculations as line / WIDTH,
  // and the 0,10,20,30,..310 component as (line % WIDTH) % 32 = (line % 32),
  // and the multiple of 320 component as (line % WIDTH) / 32

  u32 me = get_local_id(0);
  u32 WG = IN_WG;
  u32 SIZEY = WG / IN_SIZEX;

  in += line / WIDTH * WG;
  in += line % IN_SIZEX * SIZEY;
  in += line % WIDTH / IN_SIZEX * (SMALL_HEIGHT / SIZEY) * MIDDLE * WG;
  in += me / SIZEY * MIDDLE * WG + me % SIZEY;
  for (i32 i = 0; i < NH; ++i) { u[i] = in[i * G_H / SIZEY * MIDDLE * WG]; }
}

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

T fweightUnitStep(u32 i) {
  T FWEIGHTS_[] = FWEIGHTS;
  return FWEIGHTS_[i];
}

T iweightUnitStep(u32 i) {
  T IWEIGHTS_[] = IWEIGHTS;
  return IWEIGHTS_[i];
}

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
#define WSUB(i, w) u[i] = mul_by_conjugate(u[i], w);

void middleMul(T2 *u, u32 s, Trig trig, BigTab TRIG_BH) {
  assert(s < SMALL_HEIGHT);
  if (MIDDLE == 1) { return; }
  T2 w = slowTrig_BH(s, SMALL_HEIGHT, TRIG_BH);

#if MM_CHAIN == 3
// This is our slowest version - used when we are extremely worried about round off error.
// Maximum multiply chain length is 1.
  WADD(1, w);
  if ((MIDDLE - 2) % 3) {
    T2 base = slowTrig_BH(s * 2, SMALL_HEIGHT * 2, TRIG_BH);
    WADD(2, base);
    if ((MIDDLE - 2) % 3 == 2) {
      WADD(3, base);
      WADD(3, w);
    }
  }
  for (i32 i = (MIDDLE - 2) % 3 + 3; i < MIDDLE; i += 3) {
    T2 base = slowTrig_BH(s * i, SMALL_HEIGHT * i, TRIG_BH);
    WADD(i - 1, base);
    WADD(i, base);
    WADD(i + 1, base);
    WSUB(i - 1, w);
    WADD(i + 1, w);
  }

#elif MM_CHAIN == 1 || MM_CHAIN == 2

// This is our second and third fastest versions - used when we are somewhat worried about round off error.
// Maximum multiply chain length is MIDDLE/2 or MIDDLE/4.
  WADD(1, w);
  WADD(2, sq(w));
  i32 group_start, group_size;
  for (group_start = 3; group_start < MIDDLE; group_start += group_size) {
#if MM_CHAIN == 2 && MIDDLE > 4
    group_size = (group_start == 3 ? (MIDDLE - 3) / 2 : MIDDLE - group_start);
#else
    group_size = MIDDLE - 3;
#endif
    i32 midpoint = group_start + group_size / 2;
    T2 base = slowTrig_BH(s * midpoint, SMALL_HEIGHT * midpoint, TRIG_BH);
    T2 base2 = base;
    WADD(midpoint, base);
    for (i32 i = 1; i <= group_size / 2; ++i) {
      base = mul_by_conjugate(base, w);
      WADD(midpoint - i, base);
      if (i == group_size / 2 && (group_size & 1) == 0) break;
      base2 = mul(base2, w);
      WADD(midpoint + i, base2);
    }
  }

#elif MM_CHAIN == 4

  for (int i = 1; i < MIDDLE; ++i) {
    WADD(i, trig[s + (i - 1) * SMALL_HEIGHT]);
  }
  
#else  
  WADD(1, w);
  T2 base = sq(w);
  for (i32 i = 2; i < MIDDLE; ++i) {
    WADD(i, base);
    base = mul(base, w);
  }
#endif
}

void middleMul2(T2 *u, u32 x, u32 y, double factor, BigTab TRIG_BHW) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);
  T2 w = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE, TRIG_BHW);
  
#if MM2_CHAIN == 3
// This is our slowest version - used when we are extremely worried about round off error.
// Maximum multiply chain length is 1.
  if (MIDDLE % 3) {
    T2 base = slowTrig_N(x * y, ND / MIDDLE, TRIG_BHW) * factor;
    WADD(0, base);
    if (MIDDLE % 3 == 2) {
      WADD(1, base);
      WADD(1, w);
    }
  }
  for (i32 i = MIDDLE % 3 + 1; i < MIDDLE; i += 3) {
    T2 base = slowTrig_N(x * SMALL_HEIGHT * i + x * y, ND / MIDDLE * (i + 1), TRIG_BHW) * factor;
    WADD(i - 1, base);
    WADD(i, base);
    WADD(i + 1, base);
    WSUB(i - 1, w);
    WADD(i + 1, w);
  }

#elif MM2_CHAIN == 1 || MM2_CHAIN == 2

// This is our second and third fastest versions - used when we are somewhat worried about round off error.
// Maximum multiply chain length is MIDDLE/2 or MIDDLE/4.
  i32 group_size = 0;
  for (i32 group_start = 0; group_start < MIDDLE; group_start += group_size) {
#if MM2_CHAIN == 2
    group_size = (group_start == 0 ? MIDDLE / 2 : MIDDLE - group_start);
#else
    group_size = MIDDLE;
#endif
    i32 midpoint = group_start + group_size / 2;
    T2 base = slowTrig_N(x * SMALL_HEIGHT * midpoint + x * y, ND / MIDDLE * (midpoint + 1), TRIG_BHW) * factor;
    T2 base2 = base;
    WADD(midpoint, base);
    for (i32 i = 1; i <= group_size / 2; ++i) {
      base = mul_by_conjugate(base, w);
      WADD(midpoint - i, base);
      if (i == group_size / 2 && (group_size & 1) == 0) break;
      base2 = mul(base2, w);
      WADD(midpoint + i, base2);
    }
  }

#else

// This is our fastest version - used when we are not worried about round off error.
// Maximum multiply chain length equals MIDDLE.
  T2 base = slowTrig_N(x * y, ND/MIDDLE, TRIG_BHW) * factor;
  for (i32 i = 0; i < MIDDLE; ++i) {
    WADD(i, base);
    base = mul(base, w);
  }

#endif

}

#undef WADD
#undef WSUB

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

#if STATS
void updateStats(global uint *ROE, u32 posROE, float roundMax) {
  assert(roundMax >= 0);
  u32 groupRound = work_group_reduce_max(as_uint(roundMax));

  if (get_local_id(0) == 0) { atomic_max(ROE + posROE, groupRound); }
}
#endif

// For use in tailFused below

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

// This implementation compared to the original version that is no longer included in this file takes
// better advantage of the AMD OMOD (output modifier) feature.
//
// Why does this alternate implementation work?  Let t' be the conjugate of t and note that t*t' = 1.
// Now consider these lines from the original implementation (comments appear alongside):
//      b = mul_by_conjugate(b, t); 			bt'
//      X2(a, b);					a + bt', a - bt'
//      a = sq(a);					a^2 + 2abt' + (bt')^2
//      b = sq(b);					a^2 - 2abt' + (bt')^2
//      X2(a, b);					2a^2 + 2(bt')^2, 4abt'
//      b = mul(b, t);					                 4ab
// Original code is 2 complex muls, 2 complex squares, 4 complex adds
// New code is 2 complex squares, 2 complex muls, 1 complex add PLUS a complex-mul-by-2 and a complex-mul-by-4
// NOTE:  We actually, return the result divided by 2 so that our cost for the above is
// reduced to 2 complex squares, 2 complex muls, 1 complex add PLUS a complex-mul-by-2
// ALSO NOTE: the new code works just as well if the input t value is pre-squared, but the code that calls
// onePairSq can save a mul_t8 instruction by dealing with squared t values.

#define onePairSq(a, b, conjugate_t_squared) {\
  X2conjb(a, b); \
  T2 b2 = sq(b); \
  b = mul_m2(a, b); \
  a = mad_m1(b2, conjugate_t_squared, sq(a)); \
  X2conja(a, b); \
}

// From original code t = swap(base) and we need sq(conjugate(t)).  This macro computes sq(conjugate(t)) from base^2.
#define swap_squared(a) (-a)

void pairSq(u32 N, T2 *u, T2 *v, T2 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = foo_m2(conjugate(u[i]));
      v[i] = 4 * sq(conjugate(v[i]));
    } else {
      onePairSq(u[i], v[i], swap_squared(base_squared));
    }

    if (N == NH) {
      onePairSq(u[i+NH/2], v[i+NH/2], swap_squared(-base_squared));
    }

    T2 new_base_squared = mul(base_squared, U2(0, -1));
    onePairSq(u[i+NH/4], v[i+NH/4], swap_squared(new_base_squared));

    if (N == NH) {
      onePairSq(u[i+3*NH/4], v[i+3*NH/4], swap_squared(-new_base_squared));
    }
  }
}


// This implementation compared to the original version that is no longer included in this file takes
// better advantage of the AMD OMOD (output modifier) feature.
//
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
// Original code is 5 complex muls, 6 complex adds
// New code is 5 complex muls, 1 complex square, 2 complex adds PLUS two complex-mul-by-2
// NOTE:  We actually, return the original result divided by 2 so that our cost for the above is
// reduced to 5 complex muls, 1 complex square, 2 complex adds
// ALSO NOTE: the new code can be improved further (saves a complex squaring) if the t value is squared already,
// plus the caller saves a mul_t8 instruction by dealing with squared t values!

#define onePairMul(a, b, c, d, conjugate_t_squared) { \
  X2conjb(a, b); \
  X2conjb(c, d); \
  T2 tmp = mad_m1(a, c, mul(mul(b, d), conjugate_t_squared)); \
  b = mad_m1(b, c, mul(a, d)); \
  a = tmp; \
  X2conja(a, b); \
}

void pairMul(u32 N, T2 *u, T2 *v, T2 *p, T2 *q, T2 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = conjugate(foo2_m2(u[i], p[i]));
      v[i] = mul_m4(conjugate(v[i]), conjugate(q[i]));
    } else {
      onePairMul(u[i], v[i], p[i], q[i], swap_squared(base_squared));
    }

    if (N == NH) {
      onePairMul(u[i+NH/2], v[i+NH/2], p[i+NH/2], q[i+NH/2], swap_squared(-base_squared));
    }

    T2 new_base_squared = mul(base_squared, U2(0, -1));
    onePairMul(u[i+NH/4], v[i+NH/4], p[i+NH/4], q[i+NH/4], swap_squared(new_base_squared));

    if (N == NH) {
      onePairMul(u[i+3*NH/4], v[i+3*NH/4], p[i+3*NH/4], q[i+3*NH/4], swap_squared(-new_base_squared));
    }
  }
}


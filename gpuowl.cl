// Copyright Mihai Preda and George Woltman.

/* List of user-serviceable -use flags and their effects

NO_ASM : request to not use any inline __asm()

OUT_WG,OUT_SIZEX,OUT_SPACING <AMD default is 256,32,4> <nVidia default is 256,4,1 but needs testing>
IN_WG,IN_SIZEX,IN_SPACING <AMD default is 256,32,1>  <nVidia default is 256,4,1 but needs testing>

UNROLL_WIDTH <nVidia default>
NO_UNROLL_WIDTH <AMD default>

OLD_FFT5
NEW_FFT5 <default>
NEWEST_FFT5

NEW_FFT9 <default>
OLD_FFT9

DEBUG      enable asserts. Slow, but allows to verify that all asserts hold.
STATS      enable stats about roundoff distribution and carry magnitude

---- P-1 below ----

NO_P2_FUSED_TAIL                // Do not use the big kernel tailFusedMulDelta 

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
BIG_HEIGHT = SMALL_HEIGHT * MIDDLE
ND         number of dwords
NWORDS     number of words
NW
NH
G_W        "group width"
G_H        "group height"
 */

#define STR(x) XSTR(x)
#define XSTR(x) #x

#define OVERLOAD __attribute__((overloadable))

#pragma OPENCL FP_CONTRACT ON

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// 64-bit atomics used in kernel sum64
// If 64-bit atomics aren't available, sum64() can be implemented with 32-bit
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
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

typedef float T;
typedef float2 TT;
#define RE(a) (a.x)
#define IM(a) (a.y)

void bar() { barrier(0); }

TT U2(T a, T b) { return (TT) (a, b); }

OVERLOAD T sum(T a, T b) { return a + b; }

OVERLOAD T mad1(T x, T y, T z) { return x * y + z; }
  // fma(x, y, z); }

OVERLOAD T mul(T x, T y) { return x * y; }

T add1_m2(T x, T y) { return 2 * sum(x, y); }
T sub1_m2(T x, T y) { return 2 * sum(x, -y); }
T mul1_m2(T x, T y) { return 2 * mul(x, y); }

// x * (y + 1)
OVERLOAD T fancyMul(T x, const T y) { return fma(x, y, x); }

OVERLOAD TT fancyMul(TT x, const TT y) {
  return U2(fancyMul(RE(x), RE(y)), fancyMul(IM(x), IM(y)));
}

T mad1_m2(T a, T b, T c) { return 2 * mad1(a, b, c); }
T mad1_m4(T a, T b, T c) { return 4 * mad1(a, b, c); }

// complex square
OVERLOAD TT sq(TT a) { return U2(mad1(RE(a), RE(a), - IM(a) * IM(a)), mul1_m2(RE(a), IM(a))); }

// complex mul
OVERLOAD TT mul(TT a, TT b) { return U2(mad1(RE(a), RE(b), - IM(a) * IM(b)), mad1(RE(a), IM(b), IM(a) * RE(b))); }

bool test(u32 bits, u32 pos) { return (bits >> pos) & 1; }

#define STEP (NWORDS - (EXP % NWORDS))
u32 EXTRA(u64 k) { return STEP * k % NWORDS; }
bool isBigExtra(u32 extra) { return extra < NWORDS - STEP; }
bool isBigK(u32 k) { return isBigExtra(EXTRA(k)); }
u32 bitlen(bool b) { return EXP / NWORDS + b; }
u32 bitlenK(u32 k) { return bitlen(isBigK(k)); }
uint2 bitlen2K(u32 k) { return (uint2) (bitlenK(2*k), bitlenK(2*k + 1)); }

TT mul_m2(TT a, TT b) { return mul(a, b) * 2; }
TT mul_m4(TT a, TT b) { return mul(a, b) * 4; }

TT mad_m1(TT a, TT b, TT c) {
   return U2(mad1(RE(a), RE(b), mad1(IM(a), -IM(b), RE(c))), mad1(RE(a), IM(b), mad1(IM(a), RE(b), IM(c))));
}

// complex fma * 2
TT mad_m2(TT a, TT b, TT c) { return mad_m1(a, b, c) * 2; }

TT mul_t4(TT a)  { return U2(IM(a), -RE(a)); } // mul(a, U2( 0, -1)); }
TT mul_t8(TT a)  { return U2(IM(a) + RE(a), IM(a) - RE(a)) *   (T) (M_SQRT1_2); }  // mul(a, U2( 1, -1)) * (T)(M_SQRT1_2); }
TT mul_3t8(TT a) { return U2(RE(a) - IM(a), RE(a) + IM(a)) * - (T) (M_SQRT1_2); }  // mul(a, U2(-1, -1)) * (T)(M_SQRT1_2); }

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

// In a straightforward implementation, inverse weights are between 0.5 and 1.0.  We use inverse weights between 1.0 and 2.0
// because it allows us to implement this routine with a single OR instruction on the exponent.   The original implementation
// where this routine took as input values from 0.25 to 1.0 required both an AND and an OR instruction on the exponent.
// return iw <= 1.0 ? iw * 2 : iw;

float optionalDouble(float iw) {
  // FP32 exponent bias is 127. Thus 1.0f has exponent 0x7f. Setting the lowest bit of the exponent
  // will double a number from [0.5, 1) to [1, 2).
  assert(iw > 0.5f && iw < 2);
  return as_float(as_uint(iw) | 0x00800000u);
}

// In a straightforward implementation, weights are between 1.0 and 2.0.  We use weights between 2.0 and 4.0 because
// it allows us to implement this routine with a single AND instruction on the exponent.   The original implementation
// where this routine took as input values from 1.0 to 4.0 required both an AND and an OR instruction on the exponent.

float optionalHalve(float w) {
   // FP32 exponent bias is 127; so 2.0f has exponent 0x80.
   assert(w >= 2 && w < 8);
   return as_float(as_uint(w) & 0xff7fffffu);
}

#if HAS_ASM
i32  lowBits(i32 u, u32 bits) { i32 tmp; __asm("v_bfe_i32 %0, %1, 0, %2" : "=v" (tmp) : "v" (u), "v" (bits)); return tmp; }
i32 xtract32(i64 x, u32 bits) { i32 tmp; __asm("v_alignbit_b32 %0, %1, %2, %3" : "=v"(tmp) : "v"(as_int2(x).y), "v"(as_int2(x).x), "v"(bits)); return tmp; }
#else
i32  lowBits(i32 u, u32 bits) { return ((u << (32 - bits)) >> (32 - bits)); }
i32 xtract32(i64 x, u32 bits) { return ((i32) (x >> bits)); }
#endif

i64 I64(T x, T *maxROE) {
  T r = rint(x);
  *maxROE = max(*maxROE, fabs(x - r));
  return r;
}

Word carryStep64(i64 x, i64 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);

  if ((-w == (1 << (nBits - 1))) && (x > 0)) {
    w = -w;
    assert(x >= w);
    assert(((x - w) & ((1 << nBits) - 1)) == 0);
  }

  *outCarry = (x - w) >> nBits;
  return w;
}

Word carryStep32(i32 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);
  if ((-w == (1 << (nBits - 1))) && (x > 0)) { w = -w; }
  *outCarry = (x - w) >> nBits;
  return w;
}

typedef TT T2;

Word2 carryPair(T2 u, i64 *outCarry, bool b1, bool b2, i64 inCarry, T *maxROE) {
  i64 midCarry;
  Word a = carryStep64(I64(u.x, maxROE) + inCarry, &midCarry, b1);
  Word b = carryStep64(I64(u.y, maxROE) + midCarry, outCarry, b2);
  return (Word2) (a, b);
}

Word2 carryFinal(Word2 a, i64 inCarry, bool b1) {
  i64 tmpCarry;
  a.x = carryStep64(a.x + inCarry, &tmpCarry, b1);
  a.y += tmpCarry;
  return a;
}

Word2 carryPairMul(T2 u, i64 *outCarry, bool b1, bool b2, i64 inCarry, T *maxROE) {
  i64 midCarry;
  Word a = carryStep64(3 * I64(u.x, maxROE) + inCarry, &midCarry, b1);
  Word b = carryStep64(3 * I64(u.y, maxROE) + midCarry, outCarry, b2);
  return (Word2) (a, b);
}

// Carry propagation from word and carry.
Word2 carryWord64(Word2 a, i64 *carry, bool b1, bool b2) {
  a.x = carryStep64(a.x + *carry, carry, b1);
  a.y = carryStep64(a.y + *carry, carry, b2);
  return a;
}

Word2 carryWord32(Word2 a, i32 *carry, bool b1, bool b2) {
  a.x = carryStep32(a.x + *carry, carry, b1);
  a.y = carryStep32(a.y + *carry, carry, b2);
  return a;
}

// Propagate carry this many pairs of words.
#define CARRY_LEN 8

T2 addsub(T2 a) { return U2(RE(a) + IM(a), RE(a) - IM(a)); }
T2 addsub_m2(T2 a) { return addsub(a) * 2; }

// computes 2*(a.x*b.x+a.y*b.y) + i*2*(a.x*b.y+a.y*b.x)
// which happens to be the cyclical convolution (a.x, a.y)x(b.x, b.y) * 2
T2 foo2(T2 a, T2 b) {
  a = addsub(a);
  b = addsub(b);
  return addsub(U2(RE(a) * RE(b), IM(a) * IM(b)));
}

T2 foo2_m2(T2 a, T2 b) {
  return foo2(a, b) * 2;
}

// computes 2*[x^2+y^2 + i*(2*x*y)]. i.e. 2 * cyclical autoconvolution of (x, y)
T2 foo(T2 a) { return foo2(a, a); }
T2 foo_m2(T2 a) { return foo2_m2(a, a); }

#define X2(a, b) { T2 t = a; a = t + b; b = t - b; }

// Same as X2(a, b), b = mul_t4(b)
#define X2_mul_t4(a, b) { T2 t = a; a = t + b; t.x = RE(b) - t.x; RE(b) = t.y - IM(b); IM(b) = t.x; }

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
  X2(u[1], u[3]); u[3] = mul_t4(u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);
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
//#include "fft3.cl"
//#include "fft5.cl"
//#include "fft6.cl"
//#include "fft7.cl"
//#include "fft9.cl"
//#include "fft10.cl"
//#include "fft11.cl"
//#include "fft12.cl"
//#include "fft13.cl"
//#include "fft14.cl"
//#include "fft15.cl"

#include "trigSP.cl"

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
#else
typedef global const T2* Trig;
#endif

void tabMul(u32 WG, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  
  for (u32 i = 1; i < n; ++i) {
    u[i] = mul(u[i], slowTrig(i * (me & ~(f-1)), n * WG, n * WG));
    // u[i] = mul(u[i], trig[(me & ~(f-1)) + (i - 1) * WG]);
  }
}

void shuflAndMul(u32 WG, local T2 *lds, T2 *u, u32 n, u32 f) {
  tabMul(WG, u, n, f);
  shufl(WG, lds, u, n, f);
}

// 64x4
void fft256w(local T2 *lds, T2 *u) {
  UNROLL_WIDTH_CONTROL
  for (u32 s = 0; s <= 4; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul(64, lds, u, 4, 1u << s);
  }
  fft4(u);
}

void fft256h(local T2 *lds, T2 *u) {
  for (u32 s = 0; s <= 4; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul(64, lds, u, 4, 1u << s);
  }
  fft4(u);
}

// 64x8
void fft512w(local T2 *lds, T2 *u) {
  UNROLL_WIDTH_CONTROL
  for (u32 s = 0; s <= 3; s += 3) {
    if (s) { bar(); }
    fft8(u);
    shuflAndMul(64, lds, u, 8, 1u << s);
  }
  fft8(u);
}

void fft512h(local T2 *lds, T2 *u) {
  for (u32 s = 0; s <= 3; s += 3) {
    if (s) { bar(); }
    fft8(u);
    shuflAndMul(64, lds, u, 8, 1u << s);
  }
  fft8(u);
}

// 256x4
void fft1Kw(local T2 *lds, T2 *u) {
  UNROLL_WIDTH_CONTROL
  for (i32 s = 0; s <= 6; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul(256, lds, u, 4, 1u << s);
  }
  fft4(u);
}

void fft1Kh(local T2 *lds, T2 *u) {
  for (i32 s = 0; s <= 6; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul(256, lds, u, 4, 1u << s);
  }
  fft4(u);
}

// 512x8
void fft4Kw(local T2 *lds, T2 *u) {
  UNROLL_WIDTH_CONTROL
  for (u32 s = 0; s <= 6; s += 3) {
    if (s) { bar(); }
    fft8(u);
    shuflAndMul(512, lds, u, 8, 1u << s);
  }
  fft8(u);
}

void fft4Kh(local T2 *lds, T2 *u) {
  for (u32 s = 0; s <= 6; s += 3) {
    if (s) { bar(); }
    fft8(u);
    shuflAndMul(512, lds, u, 8, 1u << s);
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


// Global data

TT THREAD_WEIGHTS[G_W];
TT CARRY_WEIGHTS[BIG_HEIGHT / CARRY_LEN];

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

KERNEL(64) writeGlobals(global T2* trig2Sh, global T2* trigBh, global T2* trigN,
                        global T2* trigW,
                        global T2* threadWeights, global T2* carryWeights
                        ) {
#if 0
  for (u32 k = get_global_id(0); k < 2 * SMALL_HEIGHT/8 + 1; k += get_global_size(0)) { TRIG_2SH[k] = trig2ShDP[k]; }
  for (u32 k = get_global_id(0); k < BIG_HEIGHT/8 + 1; k += get_global_size(0))       { TRIG_BH[k] = trigBhDP[k]; }
#endif

  // Weights
  for (u32 k = get_global_id(0); k < G_W; k += get_global_size(0)) { THREAD_WEIGHTS[k] = threadWeights[k]; }
  for (u32 k = get_global_id(0); k < BIG_HEIGHT / CARRY_LEN; k += get_global_size(0)) { CARRY_WEIGHTS[k] = carryWeights[k]; }  
}

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

#define P(x) global x * restrict
#define CP(x) const P(x)

// Read 64 Word2 starting at position 'startDword'.
KERNEL(64) readResidue(P(Word2) out, CP(Word2) in, u32 startDword) {
  u32 me = get_local_id(0);
  u32 k = (startDword + me) % ND;
  u32 y = k % BIG_HEIGHT;
  u32 x = k / BIG_HEIGHT;
  out[me] = in[WIDTH * y + x];
}

u32 transPos(u32 k, u32 middle, u32 width) { return k / width + k % width * middle; }

KERNEL(256) sum64(global ulong* out, u32 sizeBytes, global ulong* in) {
  if (get_global_id(0) == 0) { out[0] = 0; }

  ulong sum = 0;
  for (i32 p = get_global_id(0); p < sizeBytes / sizeof(u64); p += get_global_size(0)) {
    sum += in[p];
  }
  sum = work_group_reduce_add(sum);
  if (get_local_id(0) == 0) { atom_add(&out[0], sum); }
}

u32 ROL31(u32 w, u32 shift) {
  return ((w << shift) & 0x7fffffff) + (w >> (31 - shift));
}

u32 modM31(u64 a) {
  a = (a & 0x7fffffff) + (a >> 31);
  u32 b = (a & 0x7fffffff) + (a >> 31);
  return  (b & 0x7fffffff) + (b >> 31);
}

#define SGW 64
KERNEL(SGW) stats(P(i64) outStats, global Word2 *A) {
  u32 me = get_local_id(0);
  u32 g = get_group_id(0);
  u32 gx = g % (WIDTH / SGW);
  u32 gy = g / (WIDTH / SGW);

  u32 sumAbs = 0;
  i32 sum = 0;
  u64 sumM31 = 0;

  const u32 NL = 16;

  u32 k = NL * gy + BIG_HEIGHT * SGW * gx + BIG_HEIGHT * me;
  assert(2 * k < NWORDS);
  u32 shift = k ? (2 * k * (u64) EXP / NWORDS + 1) % 31 : 0;

  i32 carry = 0;

  for (u32 i = 0; i < NL; ++i, ++k) {
    Word2 a = A[WIDTH * i + WIDTH * NL * gy + SGW * gx + me];
    uint2 len = bitlen2K(k);
    sum += a.x + a.y;
    sumAbs += abs(a.x) + abs(a.y);

    assert(abs(a.x) <= (1 << (len.x - 1)));
    a.x += carry;
    carry = a.x >> 31; // bool isNeg = (a.x < 0);
    u32 w1 = carry ? a.x + (1 << len.x) : a.x;
    sumM31 += ROL31(w1, shift);

    shift += len.x;
    shift = (shift >= 31) ? shift - 31 : shift;

    assert(abs(a.y) <= (1 << (len.y - 1)));
    a.y += carry;
    carry = a.y >> 31; // isNeg = (a.y < 0);
    u32 w2 = carry ? a.y + (1 << len.y) : a.y;
    sumM31 += ROL31(w2, shift);

    shift += len.y;
    shift = (shift >= 31) ? shift - 31 : shift;
  }

  const u32 MINUS_ONE = 0x7ffffffe; // (1u << 31) - 2u;
  if (carry) {
    if (k == ND) {
      --sumM31;
    } else {
      sumM31 += ROL31(MINUS_ONE, shift);
    }
  }

  i64 groupSumAbs = work_group_reduce_add((i64) sumAbs);
  i32 groupSum    = work_group_reduce_add(sum);
  i64 groupSumM31 = work_group_reduce_add((i64) sumM31);

  if (me == 0) {
    atom_add(&outStats[0], groupSumAbs);
    atom_add(&outStats[1], groupSum);
    atom_add(&outStats[2], groupSumM31);
  }
}
#undef SGW

void fft_WIDTH(local T2 *lds, T2 *u) {
#if WIDTH == 256
  fft256w(lds, u);
#elif WIDTH == 512
  fft512w(lds, u);
#elif WIDTH == 1024
  fft1Kw(lds, u);
#elif WIDTH == 4096
  fft4Kw(lds, u);
#else
#error unexpected WIDTH.  
#endif  
}

void fft_HEIGHT(local T2 *lds, T2 *u) {
#if SMALL_HEIGHT == 256
  fft256h(lds, u);
#elif SMALL_HEIGHT == 512
  fft512h(lds, u);
#elif SMALL_HEIGHT == 1024
  fft1Kh(lds, u);
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

// Do an fft_WIDTH after a transposeH (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_W) fftW(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[WIDTH / 2];
  
  T2 u[NW];
  u32 g = get_group_id(0);

  readCarryFusedLine(in, u, g);

  fft_WIDTH(lds, u);
  out += WIDTH * g;
  write(G_W, NW, u, out, 0);
}

// Do an FFT Height after a transposeW (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHin(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[SMALL_HEIGHT / 2];
  
  T2 u[NH];
  u32 g = get_group_id(0);

  readTailFusedLine(in, u, g);

  fft_HEIGHT(lds, u);

  out += SMALL_HEIGHT * transPos(g, MIDDLE, WIDTH);
  write(G_H, NH, u, out, 0);
}

// Do an FFT Height after a pointwise squaring/multiply (data is in sequential order)
KERNEL(G_H) fftHout(P(T2) io, Trig smallTrig) {
  local T2 lds[SMALL_HEIGHT / 2];
  
  T2 u[NH];
  u32 g = get_group_id(0);

  io += g * SMALL_HEIGHT;

  read(G_H, NH, u, io, 0);

  fft_HEIGHT(lds, u);
  write(G_H, NH, u, io, 0);
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

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig) {
  local T2 lds[WIDTH / 2];

  T2 u[NW];
  u32 g = get_group_id(0);

  u32 step = WIDTH * g;
  in  += step;
  out += step;

  u32 me = get_local_id(0);

  T base = optionalHalve(fancyMul(CARRY_WEIGHTS[g / CARRY_LEN].y, THREAD_WEIGHTS[me].y));
  base = optionalHalve(fancyMul(base, fweightUnitStep(g % CARRY_LEN)));

  for (u32 i = 0; i < NW; ++i) {
    T w1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    T w2 = optionalHalve(fancyMul(w1, WEIGHT_STEP));
    u32 p = G_W * i + me;
    u[i] = U2(in[p].x, in[p].y) * U2(w1, w2);
  }

  fft_WIDTH(lds, u);
  
  write(G_W, NW, u, out, 0);
}

void fft_MIDDLE(T2 *u) {
#if MIDDLE == 1
  // Do nothing
#elif MIDDLE == 2
  fft2(u);
#elif MIDDLE == 4
  fft4(u);
#elif MIDDLE == 8
  fft8(u);
/* Disable NPOT middle temporarilly
 *
#elif MIDDLE == 3
  fft3(u);
#elif MIDDLE == 5
  fft5(u);
#elif MIDDLE == 6
  fft6(u);
#elif MIDDLE == 7
  fft7(u);
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
*/
#else
#error Only power-of-two MIDDLEs enabled
#endif
}

// Apply the twiddles needed after fft_MIDDLE and before fft_HEIGHT in forward FFT.
// Also used after fft_HEIGHT and before fft_MIDDLE in inverse FFT.

void middleMul(T2 *u, u32 s) {
  assert(s < SMALL_HEIGHT);
  if (MIDDLE == 1) { return; }

  for (u32 i = 1; i < MIDDLE; ++i) {
     u[i] = mul(u[i], slowTrig_BH(s * i, SMALL_HEIGHT * i));
  }
}

void middleMul2(T2 *u, u32 x, u32 y) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);

  for (u32 i = 0; i < MIDDLE; ++i) {
    u[i] = mul(u[i], slowTrig_N((SMALL_HEIGHT * i + y) * x, ND / MIDDLE * (i + 1)));
  }
}

void middleFactor(T2 *u, T factor) {
  for (u32 i = 0; i < MIDDLE; ++i) { u[i] = u[i] * factor; }
}

// Do a partial transpose during fftMiddleIn/Out

void middleShuffle(local T2 *lds, T2 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  local T2 *p = lds + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
  for (int i = 0; i < MIDDLE; ++i) { p[i * workgroupSize] = u[i]; }
  bar();
  for (int i = 0; i < MIDDLE; ++i) { u[i] = lds[me + workgroupSize * i]; }
}

KERNEL(IN_WG) fftMiddleIn(P(T2) out, volatile CP(T2) in) {
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

  middleMul2(u, startx + mx, starty + my);

  fft_MIDDLE(u);

  middleMul(u, starty + my);

  local T2 lds[IN_WG * MIDDLE];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);

  // out += BIG_HEIGHT * startx + starty + BIG_HEIGHT * my + mx;
  // for (u32 i = 0; i < MIDDLE; ++i) { out[i * SMALL_HEIGHT] = u[i]; }
  
  out += gx * (BIG_HEIGHT * IN_SIZEX) + gy * (MIDDLE * IN_WG) + me;
  for (i32 i = 0; i < MIDDLE; ++i) { out[i * IN_WG] = u[i]; }
}

KERNEL(OUT_WG) fftMiddleOut(P(T2) out, P(T2) in, P(u32) ROE) {
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


  middleMul(u, startx + mx);

  fft_MIDDLE(u);

  middleMul2(u, starty + my, startx + mx);

  // FFT results come out multiplied by the FFT length (NWORDS).  Also, for performance reasons
  // weights and invweights are doubled meaning we need to divide by another 2^2 and 2^2.
  T factor = 1.0 / (4 * 4 * NWORDS);
  // T factor = 1.0 / (2 * NWORDS);
  middleFactor(u, factor);

  local T2 lds[OUT_WG * MIDDLE];
  middleShuffle(lds, u, OUT_WG, OUT_SIZEX);

  out += gx * (MIDDLE * WIDTH * OUT_SIZEX);
  out += (gy / OUT_SPACING) * (MIDDLE * (OUT_WG * OUT_SPACING));
  out += (gy % OUT_SPACING) * SIZEY;
  out += (me / SIZEY) * (OUT_SPACING * SIZEY);
  out += (me % SIZEY);

  for (i32 i = 0; i < MIDDLE; ++i) { out[i * (OUT_WG * OUT_SPACING)] = u[i]; }

  if (get_global_id(0) == 0) {
    u32 pos = ROE[0];
    if (pos < ROE_SIZE - 1) {
      ROE[0] = ++pos;
      ROE[pos] = 0;
    }
  }
}

void updateROE(global u32 *ROE, float roundMax) {
  assert(roundMax >= 0 && roundMax <= 0.5f);
  float groupMax = work_group_reduce_max(roundMax);

  u32 pos = ROE[0];
  assert(pos > 0 && pos < ROE_SIZE);
  if (get_local_id(0) == 0) { atomic_max(&ROE[pos], as_uint(groupMax)); }
}

float OVERLOAD roundoff(T u, T w) {
  float x = u * w;
  float r = rint(x);
  return fabs(x - r);
}

float OVERLOAD roundoff(T2 u, T2 w) { return max(roundoff(u.x, w.x), roundoff(u.y, w.y)); }

// Carry propagation with optional MUL-3, over CARRY_LEN words.
// Input arrives conjugated and inverse-weighted.

//{{ CARRYA
KERNEL(G_W) NAME(P(Word2) out, CP(T2) in, P(CarryABM) carryOut, CP(u32) bits, P(u32) ROE) {

  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;

  CarryABM carry = 0;  
  float roundMax = 0;

  // Split 32 bits into CARRY_LEN groups of 2 bits.
#define GPW (16 / CARRY_LEN)
  u32 b = bits[(G_W * g + me) / GPW] >> (me % GPW * (2 * CARRY_LEN));
#undef GPW

  T base = optionalDouble(fancyMul(CARRY_WEIGHTS[gy].x, THREAD_WEIGHTS[me].x));
  
    base = optionalDouble(fancyMul(base, iweightStep(gx)));

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;
    T w1 = i == 0 ? base : optionalDouble(fancyMul(base, iweightUnitStep(i)));
    T w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP));
    T2 x = conjugate(in[p]) * U2(w1, w2);
        
#if DO_MUL3
    out[p] = carryPairMul(x, &carry, test(b, 2 * i), test(b, 2 * i + 1), carry, &roundMax);
#else
    out[p] = carryPair(x, &carry, test(b, 2 * i), test(b, 2 * i + 1), carry, &roundMax);
#endif
  }
  carryOut[G_W * g + me] = carry;

  updateROE(ROE, roundMax);
}
//}}

//== CARRYA NAME=carryA,DO_MUL3=0
//== CARRYA NAME=carryM,DO_MUL3=1

KERNEL(G_W) carryB(P(Word2) io, CP(i64) carryIn, CP(u32) bits) {
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

  i64 carry = carryIn[WIDTH * prevLine + prevCol];

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = i * WIDTH + me;
    io[p] = carryWord64(io[p], &carry, test(b, 2 * i), test(b, 2 * i + 1));
    if (!carry) { return; }
  }
  assert(!carry);
}

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway" carry data forwarding from one group to the next.
// See tools/expand.py for the meaning of '//{{', '//}}', '//==' -- a form of macro expansion
//{{ CARRY_FUSED
KERNEL(G_W) NAME(P(T2) out, CP(T2) in, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
                 CP(u32) bits, P(u32) ROE) {
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
  

  fft_WIDTH(lds, u);

// Convert each u value into 2 words and a 32 or 64 bit carry

  Word2 wu[NW];
  T2 weights = fancyMul(CARRY_WEIGHTS[line / CARRY_LEN], THREAD_WEIGHTS[me]);
  weights = fancyMul(U2(optionalDouble(weights.x), optionalHalve(weights.y)), U2(iweightUnitStep(line % CARRY_LEN), fweightUnitStep(line % CARRY_LEN)));

  P(i64) carryShuttlePtr = carryShuttle;
  i64 carry[NW+1];

  float roundMax = 0;
  u32 carryMax = 0;
  
  // Apply the inverse weights

  T invBase = optionalDouble(weights.x);
  
  for (u32 i = 0; i < NW; ++i) {
    T invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)));
    T invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP));
    u[i] = conjugate(u[i]) * U2(invWeight1, invWeight2);
  }

  // Generate our output carries
  for (i32 i = 0; i < NW; ++i) {
#if CF_MUL
    wu[i] = carryPairMul(u[i], &carry[i], test(b, 2 * i), test(b, 2 * i + 1), 0, &roundMax);
#else
    wu[i] = carryPair(u[i], &carry[i], test(b, 2 * i), test(b, 2 * i + 1), 0, &roundMax);
#endif
  }

  // Write out our carries
  if (gr < H) {
    for (i32 i = 0; i < NW; ++i) {
      carryShuttlePtr[gr * WIDTH + me * NW + i] = carry[i];
    }

    // Signal that this group is done writing its carries
    work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    if (me == 0) {
      atomic_store((atomic_uint *) &ready[gr], 1);
    }
  }

  updateROE(ROE, roundMax);

  if (gr == 0) { return; }

  // Wait until the previous group is ready with their carries
  if (me == 0) {
    while(!atomic_load((atomic_uint *) &ready[gr - 1]));
  }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);

  // Read from the carryShuttle carries produced by the previous WIDTH row.  Rotate carries from the last WIDTH row.
  // The new carry layout lets the compiler generate global_load_dwordx4 instructions.
  if (gr < H) {
    for (i32 i = 0; i < NW; ++i) {
      carry[i] = carryShuttlePtr[(gr - 1) * WIDTH + me * NW + i];
    }
  } else {
    for (i32 i = 0; i < NW; ++i) {
      carry[i] = carryShuttlePtr[(gr - 1) * WIDTH + (me + G_W - 1) % G_W * NW + i];
    }
    if (me == 0) {
      carry[NW] = carry[NW-1];
      for (i32 i = NW-1; i; --i) { carry[i] = carry[i-1]; }
      carry[0] = carry[NW];
    }
  }

  for (i32 i = 0; i < NW; ++i) {
    wu[i] = carryFinal(wu[i], carry[i], test(b, 2 * i));
  }
  
  T base = optionalHalve(weights.y);
  
  for (u32 i = 0; i < NW; ++i) {
    T weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    T weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP));
    u[i] = U2(wu[i].x, wu[i].y) * U2(weight1, weight2);
  }

// Clear carry ready flag for next iteration

  bar();
  if (me == 0) ready[gr - 1] = 0;

// Now do the forward FFT and write results

  fft_WIDTH(lds, u);
  write(G_W, NW, u, out, WIDTH * line);
}
//}}

//== CARRY_FUSED NAME=carryFused,    CF_MUL=0
//== CARRY_FUSED NAME=carryFusedMul, CF_MUL=1

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

//{{ MULTIPLY
KERNEL(SMALL_HEIGHT / 2) NAME(P(T2) io, CP(T2) in) {
  u32 W = SMALL_HEIGHT;
  u32 H = ND / W;



  u32 line1 = get_group_id(0);
  u32 me = get_local_id(0);

  if (line1 == 0 && me == 0) {
#if MULTIPLY_DELTA
    io[0]     = foo2_m2(conjugate(io[0]), conjugate(inA[0] - inB[0]));
    io[W / 2] = conjugate(mul_m4(io[W / 2], inA[W / 2] - inB[W / 2]));
#else
    io[0]     = foo2_m2(conjugate(io[0]), conjugate(in[0]));
    io[W / 2] = conjugate(mul_m4(io[W / 2], in[W / 2]));
#endif
    return;
  }

  u32 line2 = (H - line1) % H;
  u32 g1 = transPos(line1, MIDDLE, WIDTH);
  u32 g2 = transPos(line2, MIDDLE, WIDTH);
  u32 k = g1 * W + me;
  u32 v = g2 * W + (W - 1) - me + (line1 == 0);
  T2 a = io[k];
  T2 b = io[v];
#if MULTIPLY_DELTA
  T2 c = inA[k] - inB[k];
  T2 d = inA[v] - inB[v];
#else
  T2 c = in[k];
  T2 d = in[v];
#endif
  onePairMul(a, b, c, d, swap_squared(slowTrig_N(me * H + line1, ND / 4)));
  io[k] = a;
  io[v] = b;
}
//}}

//== MULTIPLY NAME=kernelMultiply, MULTIPLY_DELTA=0

#if NO_P2_FUSED_TAIL
//== MULTIPLY NAME=kernelMultiplyDelta, MULTIPLY_DELTA=1
#endif


//{{ TAIL_SQUARE
KERNEL(G_H) NAME(P(T2) out, CP(T2) in, Trig smallTrig1, Trig smallTrig2) {
  local T2 lds[SMALL_HEIGHT / 2];

  T2 u[NH], v[NH];

  u32 W = SMALL_HEIGHT;
  u32 H = ND / W;

  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);



#if TAIL_FUSED_LOW
  read(G_H, NH, u, in, memline1 * SMALL_HEIGHT);
  read(G_H, NH, v, in, memline2 * SMALL_HEIGHT);
#else
  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  fft_HEIGHT(lds, u);
  bar();
  fft_HEIGHT(lds, v);
#endif

  u32 me = get_local_id(0);
  if (line1 == 0) {
    // Line 0 is special: it pairs with itself, offseted by 1.
    reverse(G_H, lds, u + NH/2, true);    
    pairSq(NH/2, u,   u + NH/2, slowTrig_2SH(2 * me, SMALL_HEIGHT / 2), true);
    reverse(G_H, lds, u + NH/2, true);

    // Line H/2 also pairs with itself (but without offset).
    reverse(G_H, lds, v + NH/2, false);
    pairSq(NH/2, v,   v + NH/2, slowTrig_2SH(1 + 2 * me, SMALL_HEIGHT / 2), false);
    reverse(G_H, lds, v + NH/2, false);
  } else {    
    reverseLine(G_H, lds, v);
    pairSq(NH, u, v, slowTrig_N(line1 + me * H, ND / 4), false);
    reverseLine(G_H, lds, v);
  }

  bar();
  fft_HEIGHT(lds, v);
  bar();
  fft_HEIGHT(lds, u);
  write(G_H, NH, v, out, memline2 * SMALL_HEIGHT);
  write(G_H, NH, u, out, memline1 * SMALL_HEIGHT);
}
//}}

//== TAIL_SQUARE NAME=tailFusedSquare, TAIL_FUSED_LOW=0
//== TAIL_SQUARE NAME=tailSquareLow,   TAIL_FUSED_LOW=1


//{{ TAIL_FUSED_MUL
#if MUL_2LOW
KERNEL(G_H) NAME(P(T2) out, CP(T2) in, Trig smallTrig2) {
#else
KERNEL(G_H) NAME(P(T2) out, CP(T2) in, CP(T2) a,
#if MUL_DELTA
                 CP(T2) b,
#endif
                 Trig smallTrig1, Trig smallTrig2) {
  // The arguments smallTrig1, smallTrig2 point to the same data; they are passed in as two buffers instead of one
  // in order to work-around the ROCm optimizer which would otherwise "cache" the data once read into VGPRs, leading
  // to poor occupancy.
#endif
  
  local T2 lds[SMALL_HEIGHT / 2];

  T2 u[NH], v[NH];
  T2 p[NH], q[NH];

  u32 W = SMALL_HEIGHT;
  u32 H = ND / W;

  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);
  
#if MUL_DELTA
  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  readDelta(G_H, NH, p, a, b, memline1 * SMALL_HEIGHT);
  readDelta(G_H, NH, q, a, b, memline2 * SMALL_HEIGHT);
  fft_HEIGHT(lds, u);
  bar();
  fft_HEIGHT(lds, v);
#elif MUL_LOW
  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  read(G_H, NH, p, a, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, a, memline2 * SMALL_HEIGHT);
  fft_HEIGHT(lds, u);
  bar();
  fft_HEIGHT(lds, v);
#elif MUL_2LOW
  read(G_H, NH, u, out, memline1 * SMALL_HEIGHT);
  read(G_H, NH, v, out, memline2 * SMALL_HEIGHT);
  read(G_H, NH, p, in, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, in, memline2 * SMALL_HEIGHT);
#else
  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  readTailFusedLine(a, p, line1);
  readTailFusedLine(a, q, line2);
  fft_HEIGHT(lds, u);
  bar();
  fft_HEIGHT(lds, v);
  bar();
  fft_HEIGHT(lds, p);
  bar();
  fft_HEIGHT(lds, q);
#endif

  u32 me = get_local_id(0);
  if (line1 == 0) {
    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, slowTrig_2SH(2 * me, SMALL_HEIGHT / 2), true);
    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);

    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, slowTrig_2SH(1 + 2 * me, SMALL_HEIGHT / 2), false);
    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
  } else {    
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);
    pairMul(NH, u, v, p, q, slowTrig_N(line1 + me * H, ND / 4), false);
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);
  }

  bar();
  fft_HEIGHT(lds, v);
  write(G_H, NH, v, out, memline2 * SMALL_HEIGHT);

  bar();
  fft_HEIGHT(lds, u);
  write(G_H, NH, u, out, memline1 * SMALL_HEIGHT);
}
//}}

//== TAIL_FUSED_MUL NAME=tailMulLowLow,   MUL_DELTA=0, MUL_LOW=0, MUL_2LOW=1
//== TAIL_FUSED_MUL NAME=tailFusedMulLow, MUL_DELTA=0, MUL_LOW=1, MUL_2LOW=0
//== TAIL_FUSED_MUL NAME=tailFusedMul,    MUL_DELTA=0, MUL_LOW=0, MUL_2LOW=0

#if !NO_P2_FUSED_TAIL
// equivalent to: fftHin(io, out), multiply(out, a - b), fftH(out)
//== TAIL_FUSED_MUL NAME=tailFusedMulDelta, MUL_DELTA=1, MUL_LOW=0, MUL_2LOW=0
#endif // NO_P2_FUSED_TAIL

float fastCosSP(u32 k, u32 tau) {
  float x = (-1.0f / tau) * k;
  float out;
  __asm("v_cos_f32_e32 %0, %1" : "=v"(out) : "v" (x));
  return out;
  // return cospi(2 * x);
}

float fastSinSP(u32 k, u32 tau) {
  float x = (-1.0f / tau) * k;
  float out;
  __asm("v_sin_f32_e32 %0, %1" : "=v"(out) : "v" (x));
  return out;
  // return cospi(2 * x);
}

KERNEL(64) readHwTrig(global float2* out, u32 Tau) {
  u32 id = get_global_id(0);
  out[id] = U2(fastCosSP(id, Tau), fastSinSP(id, Tau));
}
 
kernel void testKernel1(global float2* io) {
  uint me = get_local_id(0);
  float u = io[me].x;
  float w = io[me].y;
  io[me].x = roundoff(u, w);
}

kernel void testKernel2(global float2* io) {
  uint me = get_local_id(0);
  float u = io[me].x;
  float w = io[me].y;
  float x = rint(u * w);
  io[me].x = fabs(u * w - x);
}


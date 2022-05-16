// Copyright Mihai Preda and George Woltman.

/* List of user-serviceable -use flags and their effects

NO_ASM : request to not use any inline __asm()
NO_OMOD: do not use GCN output modifiers in __asm()

OUT_WG,OUT_SIZEX,OUT_SPACING <AMD default is 256,32,4> <nVidia default is 256,4,1 but needs testing>
IN_WG,IN_SIZEX,IN_SPACING <AMD default is 256,32,1>  <nVidia default is 256,4,1 but needs testing>

UNROLL_WIDTH <nVidia default>
NO_UNROLL_WIDTH <AMD default>

OLD_FFT5
NEW_FFT5 <default>
NEWEST_FFT5

NEW_FFT9 <default>
OLD_FFT9

CARRY32 <AMD default for PRP when appropriate>
CARRY64 <nVidia default>, <AMD default for PM1 when appropriate>

TRIG_COMPUTE=<n> (default 2), can be used to balance between compute and memory for trigonometrics. TRIG_COMPUTE=0 does more memory access, TRIG_COMPUTE=2 does more compute,
and TRIG_COMPUTE=1 is in between.

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

#if !HAS_ASM
// disable everything that depends on ASM
#define NO_OMOD 1
#endif

// tuning: disable
#define NO_OMOD 1

#if CARRY32 && CARRY64
#error Conflict: both CARRY32 and CARRY64 requested
#endif

#if !CARRY32 && !CARRY64
#if AMDGPU
#define CARRY32 1
#endif
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

#if HAS_ASM && !NO_OMOD && !NO_SETMODE
// turn IEEE mode and denormals off so that mul:2 and div:2 work
#define ENABLE_MUL2() { \
    __asm volatile ("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 9, 1), 0");\
    __asm volatile ("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), 7");\
}
#else
#define ENABLE_MUL2()
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

void bar() { barrier(0); }

TT U2(T a, T b) { return (TT) (a, b); }

OVERLOAD double sum(double a, double b) { return a + b; }

OVERLOAD double mad1(double x, double y, double z) { return x * y + z; }
  // fma(x, y, z); }

OVERLOAD double mul(double x, double y) { return x * y; }

T add1_m2(T x, T y) {
#if !NO_OMOD && !SP
  T tmp;
   __asm volatile("v_add_f64 %0, %1, %2 mul:2" : "=v" (tmp) : "v" (x), "v" (y));
   return tmp;
#else
   return 2 * sum(x, y);
#endif
}

T sub1_m2(T x, T y) {
#if !NO_OMOD && !SP
  T tmp;
  __asm volatile("v_add_f64 %0, %1, -%2 mul:2" : "=v" (tmp) : "v" (x), "v" (y));
  return tmp;
#else
  return 2 * sum(x, -y);
#endif
}

// x * y * 2
T mul1_m2(T x, T y) {
#if !NO_OMOD && !SP
  T tmp;
  __asm volatile("v_mul_f64 %0, %1, %2 mul:2" : "=v" (tmp) : "v" (x), "v" (y));
  return tmp;
#else
  return 2 * mul(x, y);
#endif
}


OVERLOAD T fancyMul(T x, const T y) {
  // x * (y + 1);
  return fma(x, y, x);
}

OVERLOAD TT fancyMul(TT x, const TT y) {
  return U2(fancyMul(RE(x), RE(y)), fancyMul(IM(x), IM(y)));
}

T mad1_m2(T a, T b, T c) {
#if !NO_OMOD && !SP
  double out;
  __asm volatile("v_fma_f64 %0, %1, %2, %3 mul:2" : "=v" (out) : "v" (a), "v" (b), "v" (c));
  return out;
#else
  return 2 * mad1(a, b, c);
#endif
}

T mad1_m4(T a, T b, T c) {
#if !NO_OMOD && !SP
  double out;
  __asm volatile("v_fma_f64 %0, %1, %2, %3 mul:4" : "=v" (out) : "v" (a), "v" (b), "v" (c));
  return out;
#else
  return 4 * mad1(a, b, c);
#endif
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
i32 xtract32(i64 x, u32 bits) { return ((i32) (x >> bits)); }
#endif


// We support two sizes of carry in carryFused.  A 32-bit carry halves the amount of memory used by CarryShuttle,
// but has some risks.  As FFT sizes increase and/or exponents approach the limit of an FFT size, there is a chance
// that the carry will not fit in 32-bits -- corrupting results.  That said, I did test 2000 iterations of an exponent
// just over 1 billion.  Max(abs(carry)) was 0x637225E9 which is OK (0x80000000 or more is fatal).  P-1 testing is more
// problematic as the mul-by-3 triples the carry too.

// Rounding constant: 3 * 2^51, See https://stackoverflow.com/questions/17035464
#define RNDVAL (3.0 * (1l << 51))
// Top 32-bits of RNDVAL
#define TOPVAL (0x43380000)

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

float OVERLOAD roundoff(double u, double w) {
  double x = u * w;
  // The FMA below increases the number of significant bits in the roundoff error
  double err = fma(u, w, -rint(x));
  return fabs((float) err);
}

float OVERLOAD roundoff(double2 u, double2 w) { return max(roundoff(u.x, w.x), roundoff(u.y, w.y)); }

// For CARRY32 we don't mind pollution of this value with the double exponent bits
i64 OVERLOAD doubleToLong(double x, i32 inCarry) {
  ROUNDOFF_CHECK(x);
  return as_long(x + as_double((int2) (inCarry, TOPVAL - (inCarry < 0))));
}

i64 OVERLOAD doubleToLong(double x, i64 inCarry) {
  ROUNDOFF_CHECK(x);
  int2 tmp = as_int2(inCarry);
  tmp.y += TOPVAL;
  double d = x + as_double(tmp);

  // Extend the sign from the lower 51-bits.
  // The first 51 bits (0 to 50) are clean. Bit 51 is affected by RNDVAL. Bit 52 of RNDVAL is not stored.
  // Note: if needed, we can extend the range to 52 bits instead of 51 by taking the sign from the negation
  // of bit 51 (i.e. bit 51==1 means positive, bit 51==0 means negative).
  int2 data = as_int2(d);
  data.y = lowBits(data.y, 51 - 32);
  return as_long(data);
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

// If nBits could be 20 or more we must be careful.  doubleToLong generated x as 13 bits of trash and 51-bit signed value.
// If we right shift 20 bits we will shift some of the trash into outCarry.  First we must remove the trash bits.
#if EXP / NWORDS >= 19
  *outCarry = as_int2(x << 13).y >> (nBits - 19);
#else
  *outCarry = xtract32(x, nBits);
#endif
  *outCarry += (w < 0);
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

u32 bound(i64 carry) { return min(abs(carry), 0xfffffffful); }

typedef TT T2;

//{{ carries
Word2 OVERLOAD carryPair(T2 u, iCARRY *outCarry, bool b1, bool b2, iCARRY inCarry, u32* carryMax) {
  iCARRY midCarry;
  Word a = carryStep(doubleToLong(u.x, (iCARRY) 0) + inCarry, &midCarry, b1);
  Word b = carryStep(doubleToLong(u.y, (iCARRY) 0) + midCarry, outCarry, b2);
#if STATS
  *carryMax = max(*carryMax, max(bound(midCarry), bound(*outCarry)));
#endif
  return (Word2) (a, b);
}

Word2 OVERLOAD carryFinal(Word2 u, iCARRY inCarry, bool b1) {
  i32 tmpCarry;
  u.x = carryStep(u.x + inCarry, &tmpCarry, b1);
  u.y += tmpCarry;
  return u;
}

//}}

//== carries CARRY=32
//== carries CARRY=64

Word2 OVERLOAD carryPairMul(T2 u, i64 *outCarry, bool b1, bool b2, i64 inCarry, u32* carryMax) {
  i64 midCarry;
  Word a = carryStep(3 * doubleToLong(u.x, (i64) 0) + inCarry, &midCarry, b1);
  Word b = carryStep(3 * doubleToLong(u.y, (i64) 0) + midCarry, outCarry, b2);
#if STATS
  *carryMax = max(*carryMax, max(bound(midCarry), bound(*outCarry)));
#endif
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
#else
typedef global const T2* Trig;
#endif

void tabMul(u32 WG, Trig trig, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  
  for (u32 i = 1; i < n; ++i) {
    u[i] = mul(u[i], trig[(me & ~(f-1)) + (i - 1) * WG]);
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

global double2 TRIG_2SH[SMALL_HEIGHT / 4 + 1];
global double2 TRIG_BH[BIG_HEIGHT / 8 + 1];

#if TRIG_COMPUTE == 0
global double2 TRIG_N[ND / 8 + 1];
#elif TRIG_COMPUTE == 1
global double2 TRIG_W[WIDTH / 2 + 1];
#endif

TT THREAD_WEIGHTS[G_W];
TT CARRY_WEIGHTS[BIG_HEIGHT / CARRY_LEN];

double2 tableTrig(u32 k, u32 n, u32 kBound, global double2* trigTable) {
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

float fastSinSP(u32 k, u32 tau, u32 M) {
  // These DP coefs are optimized for computing sin(2*pi*x) with x in [0, 1/8], using
  // sin(2*pi*x) = R[0]*x + R[1]*x^3 + R[2]*x^5 + R[3]*x^7
  double R[4] = {6.2831852886262363, -41.341663358481057, 81.592672353579971, -75.407767034374388};

  // We divide the DP coefficients by the corresponding power of M, and *afterwards* truncate them to SP.
  // These constants are all computed compile-time.
  float S[4] = {
      R[0] / M,
      R[1] / (((double)M)*M*M),
      R[2] / (((double)M)*M*M*M*M),
      R[3] / (((double)M)*M*M*M*M*M*M)
      };

  // Because we took the M out, we can scale k with a power of two, i.e. without loss of precision.
  assert(tau % M == 0);
  assert(((tau / M) & (tau / M - 1)) == 0); // (tau/M is a power of 2)
  float x = (-(float)k) / (tau / M);
  float z = x * x;
  return fma(x, S[0], fma(fma(S[3], z, S[2]), z, S[1]) * x * z); // 1ulp on [0, 1/8].

  // sinpi() does argument reduction and a bunch of unnecessary stuff.
  // return sinpi(2 * x);
}

float fastCosSP(u32 k, u32 tau) {
  float x = (-1.0f / tau) * k;
  
#if HAS_ASM
  // On our intended range, [0, pi/4], v_cos_f32 seems to have 2ulps precision which is good enough.
  float out;
  __asm("v_cos_f32_e32 %0, %1" : "=v"(out) : "v" (x));
  return out;  
#else
  return cospi(2 * x);
#endif
}


#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

KERNEL(64) writeGlobals(global double2* trig2ShDP, global double2* trigBhDP, global double2* trigNDP,
                        global double2* trigW,
                        global double2* threadWeights, global double2* carryWeights
                        ) {
  for (u32 k = get_global_id(0); k < 2 * SMALL_HEIGHT/8 + 1; k += get_global_size(0)) { TRIG_2SH[k] = trig2ShDP[k]; }
  for (u32 k = get_global_id(0); k < BIG_HEIGHT/8 + 1; k += get_global_size(0)) { TRIG_BH[k] = trigBhDP[k]; }

#if TRIG_COMPUTE == 0
  for (u32 k = get_global_id(0); k < ND/8 + 1; k += get_global_size(0)) { TRIG_N[k] = trigNDP[k]; }
#elif TRIG_COMPUTE == 1
  for (u32 k = get_global_id(0); k <= WIDTH/2; k += get_global_size(0)) { TRIG_W[k] = trigW[k]; }
#endif

  // Weights
  for (u32 k = get_global_id(0); k < G_W; k += get_global_size(0)) { THREAD_WEIGHTS[k] = threadWeights[k]; }
  for (u32 k = get_global_id(0); k < BIG_HEIGHT / CARRY_LEN; k += get_global_size(0)) { CARRY_WEIGHTS[k] = carryWeights[k]; }  
}

double2 slowTrig_2SH(u32 k, u32 kBound) { return tableTrig(k, 2 * SMALL_HEIGHT, kBound, TRIG_2SH); }
double2 slowTrig_BH(u32 k, u32 kBound)  { return tableTrig(k, BIG_HEIGHT, kBound, TRIG_BH); }

// Returns e^(-i * tau * k / n), (tau == 2*pi represents a full circle). So k/n is the ratio of a full circle.
// Inverse trigonometric direction is chosen as an FFT convention.
double2 slowTrig_N(u32 k, u32 kBound)   {
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
#elif TRIG_COMPUTE == 1
  u32 a = (k + WIDTH/2) / WIDTH;
  i32 b = k - a * WIDTH;
  
  double2 cs1 = TRIG_BH[a];
  double c1 = cs1.x;
  double s1 = cs1.y;
  
  double2 cs2 = TRIG_W[abs(b)];
  double c2 = cs2.x;
  double s2 = (b < 0) ? -cs2.y : cs2.y; 

  // cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
  // sin(a+b) = cos(a)sin(b) + sin(a)cos(b)
  // c2 is stored with "-1" trick to increase accuracy, so we use fma(x,y,x) for x*(y+1)
  double c = fma(-s1, s2, fma(c1, c2, c1));
  double s = fma(c1, s2, fma(s1, c2, s1));
  double2 r = (double2)(c, s);
#elif TRIG_COMPUTE == 0
  double2 r = TRIG_N[k];
#else
#error set TRIG_COMPUTE to 0, 1 or 2.
#endif

  if (flip) { r = -swap(r); }
  if (negateCos) { r.x = -r.x; }
  if (negate) { r = -r; }
  
  return r;
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

// outEqual must be "true" on entry.
KERNEL(256) isEqual(P(bool) outEqual, u32 sizeBytes, global i64 *in1, global i64 *in2) {
  for (i32 p = get_global_id(0); p < sizeBytes / sizeof(i64); p += get_global_size(0)) {
    if (in1[p] != in2[p]) {
      *outEqual = false;
      return;
    }
  }
}

// outNotZero must be "false" on entry.
KERNEL(256) isNotZero(P(bool) outNotZero, u32 sizeBytes, global i64 *in) {
  for (i32 p = get_global_id(0); p < sizeBytes / sizeof(i64); p += get_global_size(0)) {
    if (in[p] != 0) {
      *outNotZero = true;
      return;
    }
  }
}

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

// Do an fft_WIDTH after a transposeH (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_W) fftW(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[WIDTH / 2];
  
  T2 u[NW];
  u32 g = get_group_id(0);

  readCarryFusedLine(in, u, g);
  ENABLE_MUL2();
  fft_WIDTH(lds, u, smallTrig);  
  out += WIDTH * g;
  write(G_W, NW, u, out, 0);
}

// Do an FFT Height after a transposeW (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHin(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[SMALL_HEIGHT / 2];
  
  T2 u[NH];
  u32 g = get_group_id(0);

  readTailFusedLine(in, u, g);
  ENABLE_MUL2();
  fft_HEIGHT(lds, u, smallTrig);

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
  ENABLE_MUL2();
  fft_HEIGHT(lds, u, smallTrig);
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
  ENABLE_MUL2();

  fft_WIDTH(lds, u, smallTrig);
  
  write(G_W, NW, u, out, 0);
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

void middleMul(T2 *u, u32 s, Trig trig) {
  assert(s < SMALL_HEIGHT);
  if (MIDDLE == 1) { return; }

#if MM_CHAIN == 3

// This is our slowest version - used when we are extremely worried about round off error.
// Maximum multiply chain length is 1.
  T2 w = slowTrig_BH(s, SMALL_HEIGHT);
  WADD(1, w);
  if ((MIDDLE - 2) % 3) {
    T2 base = slowTrig_BH(s * 2, SMALL_HEIGHT * 2);
    WADD(2, base);
    if ((MIDDLE - 2) % 3 == 2) {
      WADD(3, base);
      WADD(3, w);
    }
  }
  for (i32 i = (MIDDLE - 2) % 3 + 3; i < MIDDLE; i += 3) {
    T2 base = slowTrig_BH(s * i, SMALL_HEIGHT * i);
    WADD(i - 1, base);
    WADD(i, base);
    WADD(i + 1, base);
    WSUB(i - 1, w);
    WADD(i + 1, w);
  }

#elif MM_CHAIN == 1 || MM_CHAIN == 2

// This is our second and third fastest versions - used when we are somewhat worried about round off error.
// Maximum multiply chain length is MIDDLE/2 or MIDDLE/4.
  T2 w = slowTrig_BH(s, SMALL_HEIGHT);
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
    T2 base = slowTrig_BH(s * midpoint, SMALL_HEIGHT * midpoint);
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
  
  T2 w = slowTrig_BH(s, SMALL_HEIGHT);
  WADD(1, w);
  T2 base = sq(w);
  for (i32 i = 2; i < MIDDLE; ++i) {
    WADD(i, base);
    base = mul(base, w);
  }
#endif
}

void middleMul2(T2 *u, u32 x, u32 y, double factor) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);

#if MM2_CHAIN == 3

// This is our slowest version - used when we are extremely worried about round off error.
// Maximum multiply chain length is 1.
  T2 w = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE);
  if (MIDDLE % 3) {
    T2 base = slowTrig_N(x * y, ND / MIDDLE) * factor;
    WADD(0, base);
    if (MIDDLE % 3 == 2) {
      WADD(1, base);
      WADD(1, w);
    }
  }
  for (i32 i = MIDDLE % 3 + 1; i < MIDDLE; i += 3) {
    T2 base = slowTrig_N(x * SMALL_HEIGHT * i + x * y, ND / MIDDLE * (i + 1)) * factor;
    WADD(i - 1, base);
    WADD(i, base);
    WADD(i + 1, base);
    WSUB(i - 1, w);
    WADD(i + 1, w);
  }

#elif MM2_CHAIN == 1 || MM2_CHAIN == 2

// This is our second and third fastest versions - used when we are somewhat worried about round off error.
// Maximum multiply chain length is MIDDLE/2 or MIDDLE/4.
  T2 w = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE);
  i32 group_start, group_size;
  for (group_start = 0; group_start < MIDDLE; group_start += group_size) {
#if MM2_CHAIN == 2
    group_size = (group_start == 0 ? MIDDLE / 2 : MIDDLE - group_start);
#else
    group_size = MIDDLE;
#endif
    i32 midpoint = group_start + group_size / 2;
    T2 base = slowTrig_N(x * SMALL_HEIGHT * midpoint + x * y, ND / MIDDLE * (midpoint + 1)) * factor;
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
  T2 w = slowTrig_N(x * SMALL_HEIGHT, ND/MIDDLE);
  T2 base = slowTrig_N(x * y, ND/MIDDLE) * factor;
  for (i32 i = 0; i < MIDDLE; ++i) {
    u[i] = mul(u[i], base);
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


KERNEL(IN_WG) fftMiddleIn(P(T2) out, volatile CP(T2) in, Trig trig) {
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

  ENABLE_MUL2();

  middleMul2(u, startx + mx, starty + my, 1);

  fft_MIDDLE(u);

  middleMul(u, starty + my, trig);
  local T lds[IN_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);

  // out += BIG_HEIGHT * startx + starty + BIG_HEIGHT * my + mx;
  // for (u32 i = 0; i < MIDDLE; ++i) { out[i * SMALL_HEIGHT] = u[i]; }
  
  out += gx * (BIG_HEIGHT * IN_SIZEX) + gy * (MIDDLE * IN_WG) + me;
  for (i32 i = 0; i < MIDDLE; ++i) { out[i * IN_WG] = u[i]; }  

  // out += gx * (MIDDLE * SMALL_HEIGHT * IN_SIZEX) + gy * (MIDDLE * IN_WG);
  // out += me;
}

KERNEL(OUT_WG) fftMiddleOut(P(T2) out, P(T2) in, Trig trig) {
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
  ENABLE_MUL2();

  middleMul(u, startx + mx, trig);

  fft_MIDDLE(u);

  // FFT results come out multiplied by the FFT length (NWORDS).  Also, for performance reasons
  // weights and invweights are doubled meaning we need to divide by another 2^2 and 2^2.
  // Finally, roundoff errors are sometimes improved if we use the next lower double precision
  // number.  This may be due to roundoff errors introduced by applying inexact TWO_TO_N_8TH weights.
  double factor = 1.0 / (4 * 4 * NWORDS);

  middleMul2(u, starty + my, startx + mx, factor);
  local T lds[OUT_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];

  middleShuffle(lds, u, OUT_WG, OUT_SIZEX);

  out += gx * (MIDDLE * WIDTH * OUT_SIZEX);
  out += (gy / OUT_SPACING) * (MIDDLE * (OUT_WG * OUT_SPACING));
  out += (gy % OUT_SPACING) * SIZEY;
  out += (me / SIZEY) * (OUT_SPACING * SIZEY);
  out += (me % SIZEY);

  for (i32 i = 0; i < MIDDLE; ++i) { out[i * (OUT_WG * OUT_SPACING)] = u[i]; }
}

void updateStats(float roundMax, u32 carryMax, global u32* roundOut, global u32* carryStats) {
  roundMax = work_group_reduce_max(roundMax);
  carryMax = work_group_reduce_max(carryMax);
  if (get_local_id(0) == 0) {
    // Roundout 0   = count(iteration)
    // Roundout 1   = max(workgroup maxerr)
    // Roundout 2   = count(workgroup)
    // Roundout 8.. = vector of iteration_maxerr

    atomic_max(&roundOut[1], as_uint(roundMax));
    atomic_max(&carryStats[4], carryMax);
    u32 oldCount = atomic_inc(&roundOut[2]);
    assert(oldCount < get_num_groups(0));
    if (oldCount == get_num_groups(0) - 1) {
      u32 roundTmp = atomic_xchg(&roundOut[1], 0);
      carryMax = atomic_xchg(&carryStats[4], 0);
      roundOut[2] = 0;
      int nPrevIt = atomic_inc(&roundOut[0]);
      if (nPrevIt < 1024 * 1024) { roundOut[8 + nPrevIt] = roundTmp; }
      
      atom_add((global ulong *) &carryStats[0], carryMax);
      atomic_max(&carryStats[2], carryMax);
      atomic_inc(&carryStats[3]);
    }
  }
}

// Carry propagation with optional MUL-3, over CARRY_LEN words.
// Input arrives conjugated and inverse-weighted.

//{{ CARRYA
KERNEL(G_W) NAME(P(Word2) out, CP(T2) in, P(CarryABM) carryOut, CP(u32) bits, P(u32) roundOut, P(u32) carryStats) {
  ENABLE_MUL2();
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;

  CarryABM carry = 0;  
  float roundMax = 0;
  u32 carryMax = 0;

  // Split 32 bits into CARRY_LEN groups of 2 bits.
#define GPW (16 / CARRY_LEN)
  u32 b = bits[(G_W * g + me) / GPW] >> (me % GPW * (2 * CARRY_LEN));
#undef GPW

  T base = optionalDouble(fancyMul(CARRY_WEIGHTS[gy].x, THREAD_WEIGHTS[me].x));
  
    base = optionalDouble(fancyMul(base, iweightStep(gx)));

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;
    double w1 = i == 0 ? base : optionalDouble(fancyMul(base, iweightUnitStep(i)));
    double w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP));
    T2 x = conjugate(in[p]) * U2(w1, w2);
    
#if STATS
    roundMax = max(roundMax, roundoff(conjugate(in[p]), U2(w1, w2)));
#endif
    
#if DO_MUL3
    out[p] = carryPairMul(x, &carry, test(b, 2 * i), test(b, 2 * i + 1), carry, &carryMax);
#else
    out[p] = carryPair(x, &carry, test(b, 2 * i), test(b, 2 * i + 1), carry, &carryMax);
#endif
  }
  carryOut[G_W * g + me] = carry;

#if STATS
  updateStats(roundMax, carryMax, roundOut, carryStats);
#endif
}
//}}

//== CARRYA NAME=carryA,DO_MUL3=0
//== CARRYA NAME=carryM,DO_MUL3=1

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

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway" carry data forwarding from one group to the next.
// See tools/expand.py for the meaning of '//{{', '//}}', '//==' -- a form of macro expansion
//{{ CARRY_FUSED
KERNEL(G_W) NAME(P(T2) out, CP(T2) in, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
                 CP(u32) bits, P(u32) roundOut, P(u32) carryStats) {
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
  
  ENABLE_MUL2();
  fft_WIDTH(lds, u, smallTrig);

// Convert each u value into 2 words and a 32 or 64 bit carry

  Word2 wu[NW];
  T2 weights = fancyMul(CARRY_WEIGHTS[line / CARRY_LEN], THREAD_WEIGHTS[me]);
  weights = fancyMul(U2(optionalDouble(weights.x), optionalHalve(weights.y)), U2(iweightUnitStep(line % CARRY_LEN), fweightUnitStep(line % CARRY_LEN)));

#if CF_MUL
  P(CFMcarry) carryShuttlePtr = (P(CFMcarry)) carryShuttle;
  CFMcarry carry[NW+1];
#else
  P(CFcarry) carryShuttlePtr = (P(CFcarry)) carryShuttle;
  CFcarry carry[NW+1];
#endif

  float roundMax = 0;
  u32 carryMax = 0;
  
  // Apply the inverse weights

  T invBase = optionalDouble(weights.x);
  
  for (u32 i = 0; i < NW; ++i) {
    T invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)));
    T invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP));

#if STATS
    roundMax = max(roundMax, roundoff(conjugate(u[i]), U2(invWeight1, invWeight2)));
#endif

    u[i] = conjugate(u[i]) * U2(invWeight1, invWeight2);
  }

  // Generate our output carries
  for (i32 i = 0; i < NW; ++i) {
#if CF_MUL    
    wu[i] = carryPairMul(u[i], &carry[i], test(b, 2 * i), test(b, 2 * i + 1), 0, &carryMax);
#else
    wu[i] = carryPair(u[i], &carry[i], test(b, 2 * i), test(b, 2 * i + 1), 0, &carryMax);
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

#if STATS
  updateStats(roundMax, carryMax, roundOut, carryStats);
#endif

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

// Clear carry ready flag for next iteration

  bar();
  if (me == 0) ready[gr - 1] = 0;

// Now do the forward FFT and write results

  fft_WIDTH(lds, u, smallTrig);
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

  ENABLE_MUL2();

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

  ENABLE_MUL2();

#if TAIL_FUSED_LOW
  read(G_H, NH, u, in, memline1 * SMALL_HEIGHT);
  read(G_H, NH, v, in, memline2 * SMALL_HEIGHT);
#else
  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  fft_HEIGHT(lds, u, smallTrig1);
  bar();
  fft_HEIGHT(lds, v, smallTrig1);
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
  fft_HEIGHT(lds, v, smallTrig2);
  bar();
  fft_HEIGHT(lds, u, smallTrig2);
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
  
  ENABLE_MUL2();
  
#if MUL_DELTA
  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  readDelta(G_H, NH, p, a, b, memline1 * SMALL_HEIGHT);
  readDelta(G_H, NH, q, a, b, memline2 * SMALL_HEIGHT);
  fft_HEIGHT(lds, u, smallTrig1);
  bar();
  fft_HEIGHT(lds, v, smallTrig1);
#elif MUL_LOW
  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  read(G_H, NH, p, a, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, a, memline2 * SMALL_HEIGHT);
  fft_HEIGHT(lds, u, smallTrig1);
  bar();
  fft_HEIGHT(lds, v, smallTrig1);
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
  fft_HEIGHT(lds, u, smallTrig1);
  bar();
  fft_HEIGHT(lds, v, smallTrig1);
  bar();
  fft_HEIGHT(lds, p, smallTrig1);
  bar();
  fft_HEIGHT(lds, q, smallTrig1);
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
  fft_HEIGHT(lds, v, smallTrig2);
  write(G_H, NH, v, out, memline2 * SMALL_HEIGHT);

  bar();
  fft_HEIGHT(lds, u, smallTrig2);
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

KERNEL(64) readHwTrig(global float2* outSH, global float2* outBH, global float2* outN) {
  for (u32 k = get_global_id(0); k <= SMALL_HEIGHT / 4; k += get_global_size(0)) {
    outSH[k] = (float2) (fastCosSP(k, SMALL_HEIGHT * 2), fastSinSP(k, SMALL_HEIGHT * 2, 1));
  }
  
  for (u32 k = get_global_id(0); k <= BIG_HEIGHT / 8; k += get_global_size(0)) {
    outBH[k] = (float2) (fastCosSP(k, BIG_HEIGHT), fastSinSP(k, BIG_HEIGHT, MIDDLE));
  }

  for (u32 k = get_global_id(0); k <= ND / 8; k += get_global_size(0)) {
    outN[k] = (float2) (fastCosSP(k, ND), fastSinSP(k, ND, MIDDLE));
  }
}
 
// Generate a small unused kernel so developers can look at how well individual macros assemble and optimize
#ifdef TEST_KERNEL

#if 0
typedef long long i128;
typedef unsigned long long u128;

u32  U32(u32 x)   { return x; }
u64  U64(u64 x)   { return x; }
u128 U128(u128 x) { return x; }
i32  I32(i32 x)   { return x; }
i64  I64(i64 x)   { return x; }
i128 I128(i128 x) { return x; }

u32 hiU32(u64 x) { return x >> 32; }
u64 hiU64(u128 x) { return x >> 64; }

#define PRIME 0xffffffff00000001

// x * 0xffffffff
u64 mulm1(u32 x) { return (U64(x) << 32) - x; }

u64 addc(u64 a, u64 b) {
  u64 s = a + b;
  return s + (U32(-1) + (s >= a));
  // return (s < a) ? s + U32(-1) : s;
}

u64 modmul(u64 a, u64 b) {
  u128 ab = U128(a) * b;
  u64 low = U64(ab);
  u64 high = ab >> 64;
  u32 hl = U32(high);
  u32 hh = high >> 32;
  u64 s = addc(low, mulm1(hl));
  u64 hhm1 = mulm1(hh);
  s = addc(s, hhm1 << 32);
  s = addc(s, mulm1(hhm1 >> 32));
  return s;
}
#endif

kernel void testKernel(global ulong* io) {
  uint me = get_local_id(0);

  ulong a = io[me];
  ulong b = io[me + 1];
  io[me] = a + b;
}

/*
kernel void testKernel(global uint* io) {
  uint me = get_local_id(0);

  uint a = io[me];
  uint b = io[me + 1];
  uint s = a + b;
  bool carry = (s < a);
  io[me] = s + carry;
}
*/

#endif


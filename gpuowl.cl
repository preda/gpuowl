// gpuOwl, an OpenCL Mersenne primality test.
// Copyright Mihai Preda and George Woltman.

// The data is organized in pairs of words in a matrix WIDTH x HEIGHT.
// The pair (a, b) is sometimes interpreted as the complex value a + i*b.
// The order of words is column-major (i.e. transposed from the usual row-major matrix order).

#pragma OPENCL FP_CONTRACT ON

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// Common type names C++ - OpenCL.
typedef uint u32;
typedef ulong u64;

// #include "shared.h"
u32 bitposToWord(u32 E, u32 N, u32 offset) { return offset * ((u64) N) / E; }
u32 wordToBitpos(u32 E, u32 N, u32 word) { return (word * ((u64) E) + (N - 1)) / N; }

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

typedef double T;
typedef double2 T2;
typedef int Word;
typedef int2 Word2;
typedef long Carry;

T2 U2(T a, T b) { return (T2)(a, b); }


#if OLD_ISBIG || !(NWORDS & (NWORDS - 1))
#define STEP (NWORDS - (EXP % NWORDS))
uint extra(uint k) { return ((ulong) STEP) * k % NWORDS; }
bool isBigWord(uint k) { return extra(k) + STEP < NWORDS; }
#else
bool isBigWord(uint k) { u64 a = FRAC * k - 1; return a > a + FRAC; }
#endif

// Number of bits for the word at pos.
uint bitlen(uint k) { return EXP / NWORDS + isBigWord(k); }

// Propagate carry this many pairs of words.
#define CARRY_LEN 16

// complex mul
T2 mul(T2 a, T2 b) { return U2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

// complex square
#ifdef ORIG_SQ
T2 sq(T2 a) { return U2((a.x + a.y) * (a.x - a.y), 2 * a.x * a.y); }		// 2 adds, 3 muls, two muls may be FMA-able later
#else
T2 sq(T2 a) { return U2(fma (a.x, a.x, -a.y*a.y), 2 * a.x * a.y); }		// 3 muls, 1 fma, one mul may be FMA-able later
#endif

T2 mul_t4(T2 a)  { return U2(a.y, -a.x); }                          // mul(a, U2( 0, -1)); }
T2 mul_t8(T2 a)  { return U2(a.y + a.x, a.y - a.x) * M_SQRT1_2; }   // mul(a, U2( 1, -1)) * (T)(M_SQRT1_2); }
T2 mul_3t8(T2 a) { return U2(a.x - a.y, a.x + a.y) * - M_SQRT1_2; } // mul(a, U2(-1, -1)) * (T)(M_SQRT1_2); }

T  shl1(T a, uint k) { return a * (1 << k); }
T2 shl(T2 a, uint k) { return U2(shl1(a.x, k), shl1(a.y, k)); }

T2 swap(T2 a) { return U2(a.y, a.x); }
T2 conjugate(T2 a) { return U2(a.x, -a.y); }

void bar() { barrier(CLK_LOCAL_MEM_FENCE); }

Word lowBits(int u, uint bits) { return (u << (32 - bits)) >> (32 - bits); }

Word carryStep(Carry x, Carry *carry, int bits) {
  x += *carry;
  Word w = lowBits(x, bits);
  *carry = (x - w) >> bits;
  return w;
}

Carry unweight(T x, T weight) { return rint(x * weight); }

Word2 unweightAndCarry(uint mul, T2 u, Carry *carry, T2 weight, uint k) {
  Word a = carryStep(mul * unweight(u.x, weight.x), carry, bitlen(2 * k + 0));
  Word b = carryStep(mul * unweight(u.y, weight.y), carry, bitlen(2 * k + 1));
  return (Word2) (a, b);
}

T2 weight(Word2 a, T2 w) { return U2(a.x, a.y) * w; }

// No carry out. The final carry is "absorbed" in the last word.
T2 carryAndWeightFinal(Word2 u, Carry carry, T2 w, uint hk) {
  Word x = carryStep(u.x, &carry, bitlen(2 * hk));
  Word y = u.y + carry;
  return weight((Word2) (x, y), w);
}

// Carry propagation from word and carry.
Word2 carryWord(Word2 a, Carry *carry, uint pos) {
  a.x = carryStep(a.x, carry, bitlen(2 * pos + 0));
  a.y = carryStep(a.y, carry, bitlen(2 * pos + 1));
  return a;
}

T2 addsub(T2 a) { return U2(a.x + a.y, a.x - a.y); }

T2 foo2(T2 a, T2 b) {
  a = addsub(a);
  b = addsub(b);
  return addsub(U2(a.x * b.x, a.y * b.y));
}

// computes 2*[x^2+y^2 + i*(2*x*y)]. Needs a name.
T2 foo(T2 a) { return foo2(a, a); }

#if !defined(ORIG_X2) && !defined(INLINE_X2) && !defined(FMA_X2)
// default to inline X2
#define INLINE_X2 1
#endif

#if ORIG_X2
// Rocm 2.4 is not generating good code with this simple original X2.  Should rocm ever be fixed, we should use this X2
// definition rather than the alternate definition.
#define X2(a, b) { T2 t = a; a = t + b; b = t - b; }
#elif FMA_X2
// Much worse latency, less parallellism, but seems to work around rocm bug where fft4 generates 18 float ops instead of 16
#define X2(a, b) { a = a + b; b.x = fma (b.x, -2.0, a.x); b.y = fma (b.y, -2.0, a.y); }
#elif INLINE_X2
// Here's hoping the inline asm tricks rocm into not generating extra f64 ops.
#define X2(a, b) { \
	T2 t = a; a = t + b; \
	__asm( "v_add_f64 %0, %1, -%2\n" : "=v" (b.x) : "v" (t.x), "v" (b.x)); \
	__asm( "v_add_f64 %0, %1, -%2\n" : "=v" (b.y) : "v" (t.y), "v" (b.y)); \
	}
#endif

#define SWAP(a, b) { T2 t = a; a = b; b = t; }

T2 fmaT2(T a, T2 b, T2 c) { return (U2 (fma (a, b.x, c.x), fma (a, b.y, c.y))); }

// Same as X2(a, b), b = mul_t4(b)
// Saves one negation
#if ORIG_X2
// Rocm 2.4 is not generating good code with simple X2 implementations.  Should rocm ever be fixed, we should use this
// definition rather than the alternate definition.
#define X2_mul_t4(a, b) { T2 t = a; a = t + b; t.x = b.x - t.x; b.x = t.y - b.y; b.y = t.x; }
#elif FMA_X2
// Much worse latency, less parallellism, but seems to generate fewer f64 ops.
#define X2_mul_t4(a, b) { double ax = a.x; a = a + b; b.x = fma (b.y, -2.0, a.y); b.y = fma (ax, -2.0, a.x); }
#else
// Here's hoping the inline asm tricks rocm into not generating extra f64 ops.
#define X2_mul_t4(a, b) { \
	T2 t = a; a = t + b; \
	__asm( "v_add_f64 %0, %1, -%2\n" : "=v" (t.x) : "v" (b.x), "v" (t.x)); \
	__asm( "v_add_f64 %0, %1, -%2\n" : "=v" (b.x) : "v" (t.y), "v" (b.y)); \
	b.y = t.x; \
	}
#endif

// a * conjugate(b)
// saves one negation
T2 mul_by_conjugate(T2 a, T2 b) { return U2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y ); }

void fft4Core(T2 *u) {
  X2(u[0], u[2]);
  X2_mul_t4(u[1], u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);
}

void fft4(T2 *u) {
  fft4Core(u);
  // revbin [0, 2, 1, 3] undo
  SWAP(u[1], u[2]);
}

#if !defined(NEWEST_FFT8) && !defined(NEW_FFT8)
#define OLD_FFT8 1
#endif

// In rocm 2.2 this is 53 f64 ops in testKernel -- one over optimal.  However, for me it is slower
// than OLD_FFT8 when it used in "real" kernels.
#if NEWEST_FFT8

// Attempt to get more FMA by delaying mul by SQRT1_2
T2 mul_t8_delayed(T2 a)  { return U2(a.y + a.x, a.y - a.x); }
#define X2_mul_3t8_delayed(a, b) { T2 t = a; a = t + b; t = b - t; b.x = t.x - t.y; b.y = t.x + t.y; }

// Like X2 but second arg needs a multiplication by SQRT1_2
#define X2_apply_SQRT1_2(a, b) { T2 t = a; \
				 a.x = fma (b.x, M_SQRT1_2, t.x); a.y = fma (b.y, M_SQRT1_2, t.y); \
				 b.x = fma (b.x, -M_SQRT1_2, t.x); b.y = fma (b.y, -M_SQRT1_2, t.y); }

void fft4Core_delayed(T2 *u) {		// Same as fft4Core except u[1] and u[3] need to be multiplied by SQRT1_2
  X2(u[0], u[2]);
  X2_mul_t4(u[1], u[3]);		// Still need to apply SQRT1_2
  X2_apply_SQRT1_2(u[0], u[1]);
  X2_apply_SQRT1_2(u[2], u[3]);
}

void fft8Core(T2 *u) {
  X2(u[0], u[4]);
  X2(u[1], u[5]);
  X2_mul_t4(u[2], u[6]);
  X2_mul_3t8_delayed(u[3], u[7]);	// u[7] needs mul by SQRT1_2
  u[5] = mul_t8_delayed(u[5]);		// u[5] needs mul by SQRT1_2

  fft4Core(u);
  fft4Core_delayed(u + 4);
}

// In rocm 2.2 this is 57 f64 ops in testKernel -- an ugly five over optimal.
#elif NEW_FFT8

// Same as X2(a, b), b = mul_3t8(b)
//#define X2_mul_3t8(a, b) { T2 t=a; a = t+b; t = b-t; t.y *= M_SQRT1_2; b.x = t.x * M_SQRT1_2 - t.y; b.y = t.x * M_SQRT1_2 + t.y; }
#define X2_mul_3t8(a, b) { T2 t=a; a = t+b; t = b-t; b.x = (t.x - t.y) * M_SQRT1_2; b.y = (t.x + t.y) * M_SQRT1_2; }

void fft8Core(T2 *u) {
  X2(u[0], u[4]);
  X2(u[1], u[5]);
  X2_mul_t4(u[2], u[6]);
  X2_mul_3t8(u[3], u[7]);
  u[5] = mul_t8(u[5]);

  fft4Core(u);
  fft4Core(u + 4);
}

// In rocm 2.2 this is 54 f64 ops in testKernel -- two over optimal.
#elif OLD_FFT8

void fft8Core(T2 *u) {
  X2(u[0], u[4]);
  X2(u[1], u[5]);   u[5] = mul_t8(u[5]);
  X2(u[2], u[6]);   u[6] = mul_t4(u[6]);
  X2(u[3], u[7]);   u[7] = mul_3t8(u[7]);
  fft4Core(u);
  fft4Core(u + 4);
}

#endif

void fft8(T2 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

void fft3(T2 *u) {
  const double SQRT3_2 = 0x1.bb67ae8584caap-1; // sin(tau/3), sqrt(3)/2, 0.86602540378443859659;
  
  X2(u[1], u[2]);
  T2 u0 = u[0];
  u[0] += u[1];
  u[1] = u0 - u[1] / 2;
  u[2] = mul_t4(u[2] * SQRT3_2);
  X2(u[1], u[2]);
}

void fft6(T2 *u) {
  const double SQRT3_2 = 0x1.bb67ae8584caap-1; // sin(tau/3), sqrt(3)/2, 0.86602540378443859659;
  
  for (int i = 0; i < 3; ++i) { X2(u[i], u[i + 3]); }
  
  u[4] = mul(u[4], U2( 0.5, -SQRT3_2));
  u[5] = mul(u[5], U2(-0.5, -SQRT3_2));
  
  fft3(u);
  fft3(u + 3);
  
  // fix order [0, 2, 4, 1, 3, 5]
  T2 tmp = u[1];
  u[1] = u[3];
  u[3] = u[4];
  u[4] = u[2];
  u[2] = tmp;
}

#if !defined(NEWEST_FFT5) && !defined(NEW_FFT5) && !defined(OLD_FFT5)
// default to new fft5
#define NEW_FFT5 1
#endif

// Adapted from: Nussbaumer, "Fast Fourier Transform and Convolution Algorithms", 5.5.4 "5-Point DFT".

// Using rocm 2.2, testKernel shows this macro generates 34 f64 ops, 30 vgprs.
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

// Using rocm 2.2, testKernel shows this macro generates 32 f64 ops, 28 vgprs.
#elif NEW_FFT5

// Above is 34 adds. prime95 does this with 32 adds
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
  const double SIN1 = 0x1.e6f0e134454ffp-1; // sin(tau/5), 0.95105651629515353118
  const double SIN2_SIN1 = 0.618033988749894848;    // sin(2*tau/5) / sin(tau/5) = .588/.951, 0.618033988749894848
  const double COS1 = 0.309016994374947424;    // cos(tau/5), 0.309016994374947424
  const double COS2 = 0.809016994374947424;    // -cos(2*tau/5), 0.809016994374947424

  X2_mul_t4(u[1], u[4]);			// (r25+ i25+),  (i25- -r25-)
  X2_mul_t4(u[2], u[3]);			// (r34+ i34+),  (i34- -r34-)

  T2 tmp25a = u[0] + COS1 * u[1];
  T2 tmp34a = u[0] - COS2 * u[1];
  u[0] = u[0] + u[1];

  T2 tmp25b = u[4] + SIN2_SIN1 * u[3];		// (i25- +.588/.951*i34-, -r25- -.588/.951*r34-)
  T2 tmp34b = SIN2_SIN1 * u[4] - u[3];		// (.588/.951*i25- -i34-, -.588/.951*r25- +r34-)

  tmp25a = tmp25a - COS2 * u[2];
  tmp34a = tmp34a + COS1 * u[2];
  u[0] = u[0] + u[2];

  u[1] = tmp25a + SIN1 * tmp25b;
  u[4] = tmp25a - SIN1 * tmp25b;
  u[2] = tmp34a + SIN1 * tmp34b;
  u[3] = tmp34a - SIN1 * tmp34b;
}

// Using rocm 2.2, testKernel shows this macro generates 32 f64 ops, 32 vgprs.
#else			// NEWEST_FFT5

void fft5(T2 *u) {
  const double SIN1 = 0x1.e6f0e134454ffp-1;		// sin(tau/5), 0.95105651629515353118
  const double SIN2_SIN1 = 0.618033988749894848;	// sin(2*tau/5) / sin(tau/5) = .588/.951, 0.618033988749894848
  const double COS1 = 0.309016994374947424;		// cos(tau/5), 0.309016994374947424
  const double COS2 = 0.809016994374947424;		// -cos(2*tau/5), 0.809016994374947424

  X2_mul_t4(u[1], u[4]);			// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[3]);			// (r3+ i3+),  (i3- -r3-)

  T2 tmp25a = fmaT2 (COS1, u[1], u[0]);
  T2 tmp34a = fmaT2 (-COS2, u[1], u[0]);
  u[0] = u[0] + u[1];

  T2 tmp25b = fmaT2 (SIN2_SIN1, u[3], u[4]);		// (i2- +.588/.951*i3-, -r2- -.588/.951*r3-)
  T2 tmp34b = fmaT2 (SIN2_SIN1, u[4], -u[3]);		// (.588/.951*i2- -i3-, -.588/.951*r2- +r3-)

  tmp25a = fmaT2 (-COS2, u[2], tmp25a);
  tmp34a = fmaT2 (COS1, u[2], tmp34a);
  u[0] = u[0] + u[2];

  u[1] = fmaT2 (SIN1, tmp25b, tmp25a);
  u[4] = fmaT2 (-SIN1, tmp25b, tmp25a);
  u[2] = fmaT2 (SIN1, tmp34b, tmp34a);
  u[3] = fmaT2 (-SIN1, tmp34b, tmp34a);
}
#endif


#if !defined(NEWEST_FFT10) && !defined(NEW_FFT10) && !defined(OLD_FFT10)
// default to old fft10
#define OLD_FFT10 1
#endif

// Using rocm 2.2, testKernel shows this macro generates 100 f64 ops, 66 vgprs (using NEW_FFT5).
#if OLD_FFT10
void fft10(T2 *u) {
  const double COS1 =  0x1.9e3779b97f4a8p-1; // cos(tau/10), 0.80901699437494745126
  const double SIN1 = -0x1.2cf2304755a5ep-1; // sin(tau/10), 0.58778525229247313710
  const double COS2 =  0x1.3c6ef372fe95p-2;  // cos(tau/5),  0.30901699437494745126
  const double SIN2 = -0x1.e6f0e134454ffp-1; // sin(tau/5),  0.95105651629515353118
  
  for (int i = 0; i < 5; ++i) { X2(u[i], u[i + 5]); }
  u[6] = mul(u[6], U2( COS1, SIN1));
  u[7] = mul(u[7], U2( COS2, SIN2));
  u[8] = mul(u[8], U2(-COS2, SIN2));
  u[9] = mul(u[9], U2(-COS1, SIN1));
  
  fft5(u);
  fft5(u + 5);

  // fix order [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
  
  SWAP(u[3], u[6]);
  T2 tmp = u[1];
  u[1] = u[5];
  u[5] = u[7];
  u[7] = u[8];
  u[8] = u[4];
  u[4] = u[2];
  u[2] = tmp;
}

// Using rocm 2.2, testKernel shows this macro generates 92 f64 ops, 72 vgprs.
#elif NEW_FFT10

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
  const double SIN1 = 0x1.e6f0e134454ffp-1; // sin(tau/5), 0.95105651629515353118
  const double SIN2_SIN1 = 0.618033988749894848;    // sin(2*tau/5) / sin(tau/5) = .588/.951, 0.618033988749894848
  const double COS1 = 0.309016994374947424;    // cos(tau/5), 0.309016994374947424
  const double COS2 = 0.809016994374947424;    // -cos(2*tau/5), 0.809016994374947424

  X2(u[0], u[5]);				// (r16+ i16+),  (r16- i16-)
  X2_mul_t4(u[1], u[6]);			// (r27+ i27+),  (i27- -r27-)
  X2_mul_t4(u[4], u[9]);			// (r510+ i510+),  (i510- -r510-)
  X2_mul_t4(u[2], u[7]);			// (r38+ i38+),  (i38-  -r38-)
  X2_mul_t4(u[3], u[8]);			// (r49+ i49+),  (i49- -r49-)

  X2_mul_t4(u[1], u[4]);			// (r27++ i27++),  (i27+- -r27+-)
  X2_mul_t4(u[6], u[9]);			// (i27-+ -r27-+), (-r27-- -i27--)
  X2_mul_t4(u[2], u[3]);			// (r38++ i38++),  (i38+- -r38+-)
  X2_mul_t4(u[7], u[8]);			// (i38-+ -r38-+), (-r38-- -i38--)

  T2 tmp39a = u[0] + COS1 * u[1];
  T2 tmp57a = u[0] - COS2 * u[1];
  u[0] = u[0] + u[1];
  T2 tmp210a = u[5] - COS2 * u[9];
  T2 tmp48a = u[5] + COS1 * u[9];
  u[5] = u[5] + u[9];

  T2 tmp39b = u[4] + SIN2_SIN1 * u[3];		// (i27+- +.588/.951*i38+-, -r27+- -.588/.951*r38+-)
  T2 tmp57b = SIN2_SIN1 * u[4] - u[3];		// (.588/.951*i27+- -i38+-, -.588/.951*r27+- +r38+-)
  T2 tmp210b = SIN2_SIN1 * u[6] + u[7];		// (.588/.951*i27-+ +i38-+, -.588/.951*r27-+ -r38-+)
  T2 tmp48b = u[6] - SIN2_SIN1 * u[7];		// (i27-+ -.588/.951*i38-+, -r27-+ +.588/.951*r38-+)

  tmp39a = tmp39a - COS2 * u[2];
  tmp57a = tmp57a + COS1 * u[2];
  u[0] = u[0] + u[2];
  tmp210a = tmp210a - COS1 * u[8];
  tmp48a = tmp48a + COS2 * u[8];
  u[5] = u[5] - u[8];

  u[2] = tmp39a + SIN1 * tmp39b;
  u[8] = tmp39a - SIN1 * tmp39b;
  u[4] = tmp57a + SIN1 * tmp57b;
  u[6] = tmp57a - SIN1 * tmp57b;
  u[1] = tmp210a + SIN1 * tmp210b;
  u[9] = tmp210a - SIN1 * tmp210b;
  u[3] = tmp48a + SIN1 * tmp48b;
  u[7] = tmp48a - SIN1 * tmp48b;
}

// Using rocm 2.2, testKernel shows this macro generates 92 f64 ops, 64 vgprs.
#else			// NEWEST_FFT10

void fft10(T2 *u) {
  const double SIN1 = 0x1.e6f0e134454ffp-1;		// sin(tau/5), 0.95105651629515353118
  const double SIN2_SIN1 = 0.618033988749894848;	// sin(2*tau/5) / sin(tau/5) = .588/.951, 0.618033988749894848
  const double COS1 = 0.309016994374947424;		// cos(tau/5), 0.309016994374947424
  const double COS2 = 0.809016994374947424;		// -cos(2*tau/5), 0.809016994374947424

  X2(u[0], u[5]);				// (r1+ i1+),  (r1-  i1-)
  X2_mul_t4(u[1], u[6]);			// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[4], u[9]);			// (r5+ i5+),  (i5- -r5-)
  X2_mul_t4(u[2], u[7]);			// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(u[3], u[8]);			// (r4+ i4+),  (i4- -r4-)

  X2_mul_t4(u[1], u[4]);			// (r2++  i2++),  (i2+- -r2+-)
  X2_mul_t4(u[6], u[9]);			// (i2-+ -r2-+), (-r2-- -i2--)
  X2_mul_t4(u[2], u[3]);			// (r3++  i3++),  (i3+- -r3+-)
  X2_mul_t4(u[7], u[8]);			// (i3-+ -r3-+), (-r3-- -i3--)

  T2 tmp39a = fmaT2 (COS1, u[1], u[0]);
  T2 tmp57a = fmaT2 (-COS2, u[1], u[0]);
  u[0] = u[0] + u[1];
  T2 tmp210a = fmaT2 (-COS2, u[9], u[5]);
  T2 tmp48a = fmaT2 (COS1, u[9], u[5]);
  u[5] = u[5] + u[9];

  T2 tmp39b = fmaT2 (SIN2_SIN1, u[3], u[4]);		// (i2+- +.588/.951*i3+-, -r2+- -.588/.951*r3+-)
  T2 tmp57b = fmaT2 (SIN2_SIN1, u[4], -u[3]);		// (.588/.951*i2+- -i3+-, -.588/.951*r2+- +r3+-)
  T2 tmp210b = fmaT2 (SIN2_SIN1, u[6], u[7]);		// (.588/.951*i2-+ +i3-+, -.588/.951*r2-+ -r3-+)
  T2 tmp48b = fmaT2 (-SIN2_SIN1, u[7], u[6]);		// (i2-+ -.588/.951*i3-+, -r2-+ +.588/.951*r3-+)

  tmp39a = fmaT2 (-COS2, u[2], tmp39a);
  tmp57a = fmaT2 (COS1, u[2], tmp57a);
  u[0] = u[0] + u[2];
  tmp210a = fmaT2 (-COS1, u[8], tmp210a);
  tmp48a = fmaT2 (COS2, u[8], tmp48a);
  u[5] = u[5] - u[8];

  u[2] = fmaT2 (SIN1, tmp39b, tmp39a);
  u[8] = fmaT2 (-SIN1, tmp39b, tmp39a);
  u[4] = fmaT2 (SIN1, tmp57b, tmp57a);
  u[6] = fmaT2 (-SIN1, tmp57b, tmp57a);
  u[1] = fmaT2 (SIN1, tmp210b, tmp210a);
  u[9] = fmaT2 (-SIN1, tmp210b, tmp210a);
  u[3] = fmaT2 (SIN1, tmp48b, tmp48a);
  u[7] = fmaT2 (-SIN1, tmp48b, tmp48a);
}
#endif

// See prime95's gwnum/zr7.mac file for more detailed explanation of the formulas below
// R1= r1     +(r2+r7)     +(r3+r6)     +(r4+r5)
// R2= r1 +.623(r2+r7) -.223(r3+r6) -.901(r4+r5)  +(.782(i2-i7) +.975(i3-i6) +.434(i4-i5))
// R7= r1 +.623(r2+r7) -.223(r3+r6) -.901(r4+r5)  -(.782(i2-i7) +.975(i3-i6) +.434(i4-i5))
// R3= r1 -.223(r2+r7) -.901(r3+r6) +.623(r4+r5)  +(.975(i2-i7) -.434(i3-i6) -.782(i4-i5))
// R6= r1 -.223(r2+r7) -.901(r3+r6) +.623(r4+r5)  -(.975(i2-i7) -.434(i3-i6) -.782(i4-i5))
// R4= r1 -.901(r2+r7) +.623(r3+r6) -.223(r4+r5)  +(.434(i2-i7) -.782(i3-i6) +.975(i4-i5))
// R5= r1 -.901(r2+r7) +.623(r3+r6) -.223(r4+r5)  -(.434(i2-i7) -.782(i3-i6) +.975(i4-i5))

// I1= i1     +(i2+i7)     +(i3+i6)     +(i4+i5)
// I2= i1 +.623(i2+i7) -.223(i3+i6) -.901(i4+i5)  -(.782(r2-r7) +.975(r3-r6) -.434(r4-r5))
// I7= i1 +.623(i2+i7) -.223(i3+i6) -.901(i4+i5)  +(.782(r2-r7) +.975(r3-r6) -.434(r4-r5))
// I3= i1 -.223(i2+i7) -.901(i3+i6) +.623(i4+i5)  -(.975(r2-r7) -.434(r3-r6) +.782(r4-r5))
// I6= i1 -.223(i2+i7) -.901(i3+i6) +.623(i4+i5)  +(.975(r2-r7) -.434(r3-r6) +.782(r4-r5))
// I4= i1 -.901(i2+i7) +.623(i3+i6) -.223(i4+i5)  -(.434(r2-r7) -.782(r3-r6) -.975(r4-r5))
// I5= i1 -.901(i2+i7) +.623(i3+i6) -.223(i4+i5)  +(.434(r2-r7) -.782(r3-r6) -.975(r4-r5))

void fft7(T2 *u) {
  const double COS1 = 0.6234898018587335305;		// cos(tau/7)
  const double COS2 = -0.2225209339563144043;		// cos(2*tau/7)
  const double COS3 = -0.9009688679024191262;		// cos(3*tau/7)
  const double SIN1 = 0.781831482468029809;		// sin(tau/7)
  const double SIN2_SIN1 = 1.2469796037174670611;	// sin(2*tau/7) / sin(tau/7) = .975/.782
  const double SIN3_SIN1 = 0.5549581320873711914;	// sin(3*tau/7) / sin(tau/7) = .434/.782

  X2_mul_t4(u[1], u[6]);			// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[5]);			// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(u[3], u[4]);			// (r4+ i4+),  (i4- -r4-)

  T2 tmp27a = fmaT2 (COS1, u[1], u[0]);
  T2 tmp36a = fmaT2 (COS2, u[1], u[0]);
  T2 tmp45a = fmaT2 (COS3, u[1], u[0]);
  u[0] = u[0] + u[1];

  tmp27a = fmaT2 (COS2, u[2], tmp27a);
  tmp36a = fmaT2 (COS3, u[2], tmp36a);
  tmp45a = fmaT2 (COS1, u[2], tmp45a);
  u[0] = u[0] + u[2];

  tmp27a = fmaT2 (COS3, u[3], tmp27a);
  tmp36a = fmaT2 (COS1, u[3], tmp36a);
  tmp45a = fmaT2 (COS2, u[3], tmp45a);
  u[0] = u[0] + u[3];

  T2 tmp27b = fmaT2 (SIN2_SIN1, u[5], u[6]);		// .975/.782
  T2 tmp36b = fmaT2 (SIN2_SIN1, u[6], -u[4]);
  T2 tmp45b = fmaT2 (SIN2_SIN1, u[4], -u[5]);

  tmp27b = fmaT2 (SIN3_SIN1, u[4], tmp27b);		// .434/.782
  tmp36b = fmaT2 (SIN3_SIN1, -u[5], tmp36b);
  tmp45b = fmaT2 (SIN3_SIN1, u[6], tmp45b);

  u[1] = fmaT2 (SIN1, tmp27b, tmp27a);
  u[6] = fmaT2 (SIN1, -tmp27b, tmp27a);
  u[2] = fmaT2 (SIN1, tmp36b, tmp36a);
  u[5] = fmaT2 (SIN1, -tmp36b, tmp36a);
  u[3] = fmaT2 (SIN1, tmp45b, tmp45a);
  u[4] = fmaT2 (SIN1, -tmp45b, tmp45a);
}

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

  X2_mul_t4(u[1], u[10]);			// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[9]);			// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(u[3], u[8]);			// (r4+ i4+),  (i4- -r4-)
  X2_mul_t4(u[4], u[7]);			// (r5+ i5+),  (i5- -r5-)
  X2_mul_t4(u[5], u[6]);			// (r6+ i6+),  (i6- -r6-)

  T2 tmp211a = fmaT2 (COS1, u[1], u[0]);
  T2 tmp310a = fmaT2 (COS2, u[1], u[0]);
  T2 tmp49a = fmaT2 (COS3, u[1], u[0]);
  T2 tmp58a = fmaT2 (COS4, u[1], u[0]);
  T2 tmp67a = fmaT2 (COS5, u[1], u[0]);
  u[0] = u[0] + u[1];

  tmp211a = fmaT2 (COS2, u[2], tmp211a);
  tmp310a = fmaT2 (COS4, u[2], tmp310a);
  tmp49a = fmaT2 (COS5, u[2], tmp49a);
  tmp58a = fmaT2 (COS3, u[2], tmp58a);
  tmp67a = fmaT2 (COS1, u[2], tmp67a);
  u[0] = u[0] + u[2];

  tmp211a = fmaT2 (COS3, u[3], tmp211a);
  tmp310a = fmaT2 (COS5, u[3], tmp310a);
  tmp49a = fmaT2 (COS2, u[3], tmp49a);
  tmp58a = fmaT2 (COS1, u[3], tmp58a);
  tmp67a = fmaT2 (COS4, u[3], tmp67a);
  u[0] = u[0] + u[3];

  tmp211a = fmaT2 (COS4, u[4], tmp211a);
  tmp310a = fmaT2 (COS3, u[4], tmp310a);
  tmp49a = fmaT2 (COS1, u[4], tmp49a);
  tmp58a = fmaT2 (COS5, u[4], tmp58a);
  tmp67a = fmaT2 (COS2, u[4], tmp67a);
  u[0] = u[0] + u[4];

  tmp211a = fmaT2 (COS5, u[5], tmp211a);
  tmp310a = fmaT2 (COS1, u[5], tmp310a);
  tmp49a = fmaT2 (COS4, u[5], tmp49a);
  tmp58a = fmaT2 (COS2, u[5], tmp58a);
  tmp67a = fmaT2 (COS3, u[5], tmp67a);
  u[0] = u[0] + u[5];

  T2 tmp211b = fmaT2 (SIN2_SIN1, u[9], u[10]);		// .910/.541
  T2 tmp310b = fmaT2 (SIN2_SIN1, u[10], -u[6]);
  T2 tmp49b = fmaT2 (SIN2_SIN1, -u[8], u[7]);
  T2 tmp58b = fmaT2 (SIN2_SIN1, -u[6], u[8]);
  T2 tmp67b = fmaT2 (SIN2_SIN1, -u[7], -u[9]);

  tmp211b = fmaT2 (SIN3_SIN1, u[8], tmp211b);		// .990/.541
  tmp310b = fmaT2 (SIN3_SIN1, -u[7], tmp310b);
  tmp49b = fmaT2 (SIN3_SIN1, u[10], tmp49b);
  tmp58b = fmaT2 (SIN3_SIN1, -u[9], tmp58b);
  tmp67b = fmaT2 (SIN3_SIN1, u[6], tmp67b);

  tmp211b = fmaT2 (SIN4_SIN1, u[7], tmp211b);		// .756/.541
  tmp310b = fmaT2 (SIN4_SIN1, u[9], tmp310b);
  tmp49b = fmaT2 (SIN4_SIN1, u[6], tmp49b);
  tmp58b = fmaT2 (SIN4_SIN1, u[10], tmp58b);
  tmp67b = fmaT2 (SIN4_SIN1, u[8], tmp67b);

  tmp211b = fmaT2 (SIN5_SIN1, u[6], tmp211b);		// .282/.541
  tmp310b = fmaT2 (SIN5_SIN1, -u[8], tmp310b);
  tmp49b = fmaT2 (SIN5_SIN1, -u[9], tmp49b);
  tmp58b = fmaT2 (SIN5_SIN1, u[7], tmp58b);
  tmp67b = fmaT2 (SIN5_SIN1, u[10], tmp67b);

  u[1] = fmaT2 (SIN1, tmp211b, tmp211a);
  u[10] = fmaT2 (SIN1, -tmp211b, tmp211a);
  u[2] = fmaT2 (SIN1, tmp310b, tmp310a);
  u[9] = fmaT2 (SIN1, -tmp310b, tmp310a);
  u[3] = fmaT2 (SIN1, tmp49b, tmp49a);
  u[8] = fmaT2 (SIN1, -tmp49b, tmp49a);
  u[4] = fmaT2 (SIN1, tmp58b, tmp58a);
  u[7] = fmaT2 (SIN1, -tmp58b, tmp58a);
  u[5] = fmaT2 (SIN1, tmp67b, tmp67a);
  u[6] = fmaT2 (SIN1, -tmp67b, tmp67a);
}


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

  T2 tmp26812b = fmaT2 (COS1, u[7], u[9]);
  T2 tmp410b = u[9] - u[7];

  T2 tmp26812a = fmaT2 (-COS1, u[10], u[6]);
  T2 tmp410a = u[6] + u[10];

  T2 tmp212b = fmaT2 (SIN1, u[8], tmp26812b);
  T2 tmp68b = fmaT2 (-SIN1, u[8], tmp26812b);

  T2 tmp212a = fmaT2 (-SIN1, u[11], tmp26812a);
  T2 tmp68a = fmaT2 (SIN1, u[11], tmp26812a);

  T2 tmp311 = fmaT2 (-COS1, u[1], u[3]);
  u[6] = u[3] + u[1];

  T2 tmp59 = fmaT2 (-COS1, u[2], u[0]);
  u[0] = u[0] + u[2];

  u[3] = tmp410a - tmp410b;
  u[9] = tmp410a + tmp410b;
  u[1] = tmp212a + tmp212b;
  u[11] = tmp212a - tmp212b;

  u[2] = fmaT2 (SIN1, u[4], tmp311);
  u[10] = fmaT2 (-SIN1, u[4], tmp311);
  u[4] = fmaT2 (-SIN1, u[5], tmp59);
  u[8] = fmaT2 (SIN1, u[5], tmp59);

  u[5] = tmp68a + tmp68b;
  u[7] = tmp68a - tmp68b;
}

void shufl(uint WG, local T *lds, T2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  uint m = me / f;
  
  for (int b = 0; b < 2; ++b) {
    bar();
    for (uint i = 0; i < n; ++i) { lds[(m + i * WG / f) / n * f + m % n * WG + me % f] = ((T *) (u + i))[b]; }
    bar();
    for (uint i = 0; i < n; ++i) { ((T *) (u + i))[b] = lds[i * WG + me]; }
  }
}

void tabMul(uint WG, const global T2 *trig, T2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  for (int i = 1; i < n; ++i) { u[i] = mul(u[i], trig[me / f + i * (WG / f)]); }
}

void shuflAndMul(uint WG, local T *lds, const global T2 *trig, T2 *u, uint n, uint f) {
#if 0
  uint me = get_local_id(0);
  uint m = me / f;
  
  for (int b = 0; b < 2; ++b) {
    bar();
    for (uint i = 0; i < n; ++i) { lds[(m + i * WG / f) / n * f + m % n * WG + me % f] = ((T *) (u + i))[b]; }
    bar();
    for (uint i = 0; i < n; ++i) { ((T *) (u + i))[b] = lds[i * WG + me]; }
  }

  for (int i = 1; i < n; ++i) { u[i] = mul(u[i], trig[me / f + i * (WG / f)]); }  
#else
  shufl(WG, lds, u, n, f);
  tabMul(WG, trig, u, n, f);
#endif
}

// 8x8
void fft64(local T *lds, T2 *u, const global T2 *trig) {
  fft8(u);
  shuflAndMul(8, lds, trig, u, 8, 1);
  fft8(u);
}

// 64x4
void fft256(local T *lds, T2 *u, const global T2 *trig) {
  for (int s = 4; s >= 0; s -= 2) {
    fft4(u);
    shuflAndMul(64, lds, trig, u, 4, 1 << s);
  }
  fft4(u);
}

// 64x8
void fft512(local T *lds, T2 *u, const global T2 *trig) {
  for (int s = 3; s >= 0; s -= 3) {
    fft8(u);
    shuflAndMul(64, lds, trig, u, 8, 1 << s);
  }
  fft8(u);
}

// 256x4
void fft1K(local T *lds, T2 *u, const global T2 *trig) {
  for (int s = 6; s >= 0; s -= 2) {
    fft4(u);
    shuflAndMul(256, lds, trig, u, 4, 1 << s);
  }
  fft4(u);
}

// 512x8
void fft4K(local T *lds, T2 *u, const global T2 *trig) {
  for (int s = 6; s >= 0; s -= 3) {
    fft8(u);
    shuflAndMul(512, lds, trig, u, 8, 1 << s);
  }
  fft8(u);
}

// 256x8
void fft2K(local T *lds, T2 *u, const global T2 *trig) {
  for (int s = 5; s >= 2; s -= 3) {
    fft8(u);
    shuflAndMul(256, lds, trig, u, 8, 1 << s);
  }

  fft8(u);

  uint me = get_local_id(0);
  for (int b = 0; b < 2; ++b) {
    bar();
    for (int i = 0; i < 8; ++i) { lds[(me + i * 256) / 4 + me % 4 * 512] = ((T *) (u + i))[b]; }
    bar();
    for (int i = 0; i < 4; ++i) {
      ((T *) (u + i))[b]     = lds[i * 512       + me];
      ((T *) (u + i + 4))[b] = lds[i * 512 + 256 + me];
    }
  }

  for (int i = 1; i < 4; ++i) {
    u[i]     = mul(u[i],     trig[i * 512       + me]);
    u[i + 4] = mul(u[i + 4], trig[i * 512 + 256 + me]);
  }

  fft4(u);
  fft4(u + 4);

  // fix order: interleave u[0:3] and u[4:7], like (u.even, u.odd) = (u.lo, u.hi).
  SWAP(u[1], u[2]);
  SWAP(u[1], u[4]);
  SWAP(u[5], u[6]);
  SWAP(u[3], u[6]);
}

void read(uint WG, uint N, T2 *u, const global T2 *in, uint base) {
  for (int i = 0; i < N; ++i) { u[i] = in[base + i * WG + (uint) get_local_id(0)]; }
}

void write(uint WG, uint N, T2 *u, global T2 *out, uint base) {
  for (int i = 0; i < N; ++i) { out[base + i * WG + (uint) get_local_id(0)] = u[i]; }
}

void readDelta(uint WG, uint N, T2 *u, const global T2 *a, const global T2 *b, uint base) {
  for (uint i = 0; i < N; ++i) {
    uint pos = base + i * WG + (uint) get_local_id(0); 
    u[i] = a[pos] - b[pos];
  }
}

// Returns e^(-i * pi * k/n);
double2 slowTrig(int k, int n) {
  double c;
  double s = sincos(M_PI / n * k, &c);
  return U2(c, -s);
}

// transpose LDS 64 x 64.
void transposeLDS(local T *lds, T2 *u) {
  uint me = get_local_id(0);
  for (int b = 0; b < 2; ++b) {
    if (b) { bar(); }
    for (int i = 0; i < 16; ++i) {
      uint l = i * 4 + me / 64;
      lds[l * 64 + (me + l) % 64 ] = ((T *)(u + i))[b];
    }
    bar();
    for (int i = 0; i < 16; ++i) {
      uint c = i * 4 + me / 64;
      uint l = me % 64;
      ((T *)(u + i))[b] = lds[l * 64 + (c + l) % 64];
    }
  }
}

// Transpose the matrix of WxH, and MUL with FFT twiddles; by blocks of 64x64.
void transpose(uint W, uint H, local T *lds, const T2 *in, T2 *out) {
  uint GPW = W / 64, GPH = H / 64;
  
  uint g = get_group_id(0);
  uint gy = g % GPH;
  uint gx = g / GPH;
  gx = (gy + gx) % GPW;

  in   += gy * 64 * W + gx * 64;
  out  += gy * 64     + gx * 64 * H;
  
  uint me = get_local_id(0), mx = me % 64, my = me / 64;
  T2 u[16];

  for (int i = 0; i < 16; ++i) { u[i] = in[(4 * i + my) * W + mx]; }

  transposeLDS(lds, u);

  uint col = 64 * gy + mx;
  T2 base = slowTrig(col * (64 * gx + my),  W * H / 2);
  T2 step = slowTrig(col, W * H / 8);
                     
  for (int i = 0; i < 16; ++i) {
    out[(4 * i + my) * H + mx] = mul(u[i], base);
    base = mul(base, step);
  }
}

void transposeWords(uint W, uint H, local Word2 *lds, const Word2 *in, Word2 *out) {
  uint GPW = W / 64, GPH = H / 64;

  uint g = get_group_id(0);
  uint gy = g % GPH;
  uint gx = g / GPH;
  gx = (gy + gx) % GPW;

  in   += gy * 64 * W + gx * 64;
  out  += gy * 64     + gx * 64 * H;
  
  uint me = get_local_id(0);
  uint mx = me % 64;
  uint my = me / 64;
  
  Word2 u[16];

  for (int i = 0; i < 16; ++i) { u[i] = in[(4 * i + my) * W + mx]; }

  for (int i = 0; i < 16; ++i) {
    uint l = i * 4 + me / 64;
    lds[l * 64 + (me + l) % 64 ] = u[i];
  }
  bar();
  for (int i = 0; i < 16; ++i) {
    uint c = i * 4 + me / 64;
    uint l = me % 64;
    u[i] = lds[l * 64 + (c + l) % 64];
  }

  for (int i = 0; i < 16; ++i) {
    out[(4 * i + my) * H + mx] = u[i];
  }
}

#ifndef ALT_RESTRICT

#define P(x) global x * restrict
#define CP(x) const P(x)
typedef CP(T2) Trig;

#else

#define P(x) global x *
#define CP(x) const P(x)
typedef CP(T2) restrict Trig;

#endif

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

// Read 64 Word2 starting at position 'startDword'.
KERNEL(64) readResidue(CP(Word2) in, P(Word2) out, uint startDword) {
  uint me = get_local_id(0);
  uint k = (startDword + me) % ND;
  uint y = k % BIG_HEIGHT;
  uint x = k / BIG_HEIGHT;
  out[me] = in[WIDTH * y + x];
}

uint dwordToBitpos(uint dword)  { return wordToBitpos(EXP, ND, dword); }
uint bitposToDword(uint bitpos) { return bitposToWord(EXP, ND, bitpos); }
uint transPos(uint k, uint width, uint height) { return k / height + k % height * width; }

uint kAt(uint gx, uint gy, uint i) {
  return CARRY_LEN * gy + BIG_HEIGHT * G_W * gx + BIG_HEIGHT * ((uint) get_local_id(0)) + i;
}

// outEqual must be "true" on entry.
KERNEL(256) isEqual(uint sizeBytes, global long *in1, global long *in2, P(bool) outEqual) {
  for (int p = get_global_id(0); p < sizeBytes / sizeof(long); p += get_global_size(0)) {
    if (in1[p] != in2[p]) {
      *outEqual = false;
      return;
    }
  }
}

// outNotZero must be "false" on entry.
KERNEL(256) isNotZero(uint sizeBytes, global long *in, P(bool) outNotZero) {
  for (int p = get_global_id(0); p < sizeBytes / sizeof(long); p += get_global_size(0)) {
    if (in[p] != 0) {
      *outNotZero = true;
      return;
    }
  }
}

void fft_WIDTH(local T *lds, T2 *u, Trig trig) {
#if   WIDTH == 64
  fft64(lds, u, trig);
#elif WIDTH == 256
  fft256(lds, u, trig);
#elif WIDTH == 512
  fft512(lds, u, trig);
#elif WIDTH == 1024
  fft1K(lds, u, trig);
#elif WIDTH == 2048
  fft2K(lds, u, trig);
#elif WIDTH == 4096
  fft4K(lds, u, trig);
#else
#error unexpected WIDTH.  
#endif  
}

void fft_HEIGHT(local T *lds, T2 *u, Trig trig) {
#if SMALL_HEIGHT == 64
  fft64(lds, u, trig);
#elif SMALL_HEIGHT == 256
  fft256(lds, u, trig);
#elif SMALL_HEIGHT == 512
  fft512(lds, u, trig);
#elif SMALL_HEIGHT == 1024
  fft1K(lds, u, trig);
#elif SMALL_HEIGHT == 2048
  fft2K(lds, u, trig);
#else
#error unexpected SMALL_HEIGHT.
#endif
}

KERNEL(G_W) fftW(P(T2) io, Trig smallTrig) {
  local T lds[WIDTH];
  T2 u[NW];

  uint g = get_group_id(0);
  io += WIDTH * g;

  read(G_W, NW, u, io, 0);
  fft_WIDTH(lds, u, smallTrig);  
  write(G_W, NW, u, io, 0);
}

KERNEL(G_H) fftH(P(T2) io, Trig smallTrig) {
  local T lds[SMALL_HEIGHT];
  T2 u[NH];

  uint g = get_group_id(0);
  io += SMALL_HEIGHT * transPos(g, MIDDLE, WIDTH);

  read(G_H, NH, u, io, 0);
  fft_HEIGHT(lds, u, smallTrig);
  write(G_H, NH, u, io, 0);
}

// fftPremul: weight words with "A" (for IBDWT) followed by FFT.
KERNEL(G_W) fftP(CP(Word2) in, P(T2) out, CP(T2) A, Trig smallTrig) {
  local T lds[WIDTH];
  T2 u[NW];

  uint g = get_group_id(0);
  uint step = WIDTH * g;
  A   += step;
  in  += step;
  out += step;

  uint me = get_local_id(0);

  for (int i = 0; i < NW; ++i) {
    uint p = G_W * i + me;
    // u32 hk = g + BIG_HEIGHT * p;
    u[i] = weight(in[p], A[p]);
  }

  fft_WIDTH(lds, u, smallTrig);
  
  write(G_W, NW, u, out, 0);
}

void middleMul(T2 *u, uint gx, uint me) {
  T2 step = slowTrig(256 * gx + me, BIG_HEIGHT / 2);
  T2 t = step;
  for (int i = 1; i < MIDDLE; ++i, t = mul(t, step)) { u[i] = mul(u[i], t); }
}

void fft_MIDDLE(T2 *u) {
#if   MIDDLE == 3
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
#elif MIDDLE != 1
#error
#endif
}

KERNEL(256) fftMiddleIn(P(T2) io) {
  T2 u[MIDDLE];
  uint N = SMALL_HEIGHT / 256;
  uint g = get_group_id(0);
  uint gx = g % N;
  uint gy = g / N;
  uint me = get_local_id(0);
  io += BIG_HEIGHT * gy + 256 * gx;
  read(SMALL_HEIGHT, MIDDLE, u, io, 0);

  fft_MIDDLE(u);
    
  middleMul(u, gx, me);
  
  write(SMALL_HEIGHT, MIDDLE, u, io, 0);
}

KERNEL(256) fftMiddleOut(P(T2) io) {
  T2 u[MIDDLE];
  uint N = SMALL_HEIGHT / 256;
  uint g = get_group_id(0);
  uint gx = g % N;
  uint gy = g / N;
  uint me = get_local_id(0);
  io += BIG_HEIGHT * gy + 256 * gx;
  read(SMALL_HEIGHT, MIDDLE, u, io, 0);
  
  middleMul(u, gx, me);

  fft_MIDDLE(u);

  write(SMALL_HEIGHT, MIDDLE, u, io, 0);
}

// Carry propagation with optional MUL-3, over CARRY_LEN words.
// Input is conjugated and inverse-weighted.
void carryACore(uint mul, const global T2 *in, const global T2 *A, global Word2 *out, global Carry *carryOut) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);
  uint gx = g % NW;
  uint gy = g / NW;

  Carry carry = 0;

  for (int i = 0; i < CARRY_LEN; ++i) {
    uint p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;
    uint k = kAt(gx, gy, i);
    out[p] = unweightAndCarry(mul, conjugate(in[p]), &carry, A[p], k);
  }
  carryOut[G_W * g + me] = carry;
}

KERNEL(G_W) carryA(CP(T2) in, P(Word2) out, P(Carry) carryOut, CP(T2) A) {
  carryACore(1, in, A, out, carryOut);
}

KERNEL(G_W) carryM(CP(T2) in, P(Word2) out, P(Carry) carryOut, CP(T2) A) {
  carryACore(3, in, A, out, carryOut);
}

KERNEL(G_W) carryB(P(Word2) io, CP(Carry) carryIn) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);  
  uint gx = g % NW;
  uint gy = g / NW;
  
  uint step = G_W * gx + WIDTH * CARRY_LEN * gy;
  io += step;

  uint HB = BIG_HEIGHT / CARRY_LEN;

  uint prev = (gy + HB * G_W * gx + HB * me + (HB * WIDTH - 1)) % (HB * WIDTH);
  uint prevLine = prev % HB;
  uint prevCol  = prev / HB;
  Carry carry = carryIn[WIDTH * prevLine + prevCol];
  
  for (int i = 0; i < CARRY_LEN; ++i) {
    uint k = kAt(gx, gy, i);
    uint p = i * WIDTH + me;
    io[p] = carryWord(io[p], &carry, k);
    if (!carry) { return; }
  }
}

void release() {
#if 0
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
  work_group_barrier(0);
#else
  work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
#endif
}

void acquire() {
#if 0
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
  work_group_barrier(0);
#else
  work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
#endif
}

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway" carry data forwarding from one group to the next.
KERNEL(G_W) carryFused(P(T2) io, P(Carry) carryShuttle, P(uint) ready,
                       CP(T2) iA, Trig smallTrig, CP(T) groupWeights, CP(T) threadWeights) {
  local T lds[WIDTH];

  uint gr = get_group_id(0);
  uint me = get_local_id(0);
  
  uint H = BIG_HEIGHT;
  uint line = gr % H;
  uint step = WIDTH * line;
  io += step;
  // A  += step;
  iA += step;
  
  T2 u[NW];
  Word2 wu[NW];
  
  read(G_W, NW, u, io, 0);

  fft_WIDTH(lds, u, smallTrig);
  
  for (int i = 0; i < NW; ++i) {
    uint p = i * G_W + me;
    Carry carry = 0;
    uint k = line + BIG_HEIGHT * p;
    wu[i] = unweightAndCarry(1, conjugate(u[i]), &carry, iA[p], k);
    if (gr < H) { carryShuttle[gr * WIDTH + p] = carry; }
  }

  release();
  
  // Signal that this group is done writing the carry.
  if (gr < H && me == 0) {
    atomic_store_explicit((atomic_uint *) &ready[gr], 1, memory_order_release, memory_scope_device); 
  }

  if (gr == 0) { return; }

  T weight = groupWeights[line] * threadWeights[me];
  
  // Wait until the previous group is ready with the carry.
  if (me == 0) {
    while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_acquire, memory_scope_device));
    ready[gr - 1] = 0;
  }

  acquire();

  for (int i = 0; i < NW; ++i) {
    if (weight >= 2) { weight *= 0.5; }
    T weight2 = weight * WEIGHT_STEP;
    if (weight2 >= 2) { weight2 *= 0.5; }
    
    uint p = i * G_W + me;
    uint k = line + BIG_HEIGHT * p;
    u[i] = carryAndWeightFinal(wu[i], carryShuttle[(gr - 1) * WIDTH + ((p + WIDTH - gr / H) % WIDTH)], U2(weight, weight2), k);
    weight *= WEIGHT_BIGSTEP;
  }

  fft_WIDTH(lds, u, smallTrig);

  write(G_W, NW, u, io, 0);
}

// copy of carryFused() above, with the only difference the mul-by-3 in unweightAndCarry().
KERNEL(G_W) carryFusedMul(P(T2) io, P(Carry) carryShuttle, P(uint) ready,
                       CP(T2) A, CP(T2) iA, Trig smallTrig) {
  local T lds[WIDTH];

  uint gr = get_group_id(0);
  uint me = get_local_id(0);
  
  uint H = BIG_HEIGHT;
  uint line = gr % H;
  uint step = WIDTH * line;
  io += step;
  A  += step;
  iA += step;
  
  T2 u[NW];
  Word2 wu[NW];
  
  read(G_W, NW, u, io, 0);

  fft_WIDTH(lds, u, smallTrig);
  
  for (int i = 0; i < NW; ++i) {
    uint p = i * G_W + me;
    Carry carry = 0;
    uint k = line + BIG_HEIGHT * p;
    wu[i] = unweightAndCarry(3,   conjugate(u[i]), &carry, iA[p], k);
    if (gr < H) { carryShuttle[gr * WIDTH + p] = carry; }
  }

  release();
  
  // Signal that this group is done writing the carry.
  if (gr < H && me == 0) {
    atomic_store_explicit((atomic_uint *) &ready[gr], 1, memory_order_release, memory_scope_device); 
  }

  if (gr == 0) { return; }
    
  // Wait until the previous group is ready with the carry.
  if (me == 0) {
    while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_acquire, memory_scope_device));
    ready[gr - 1] = 0;
  }

  acquire();
  
  for (int i = 0; i < NW; ++i) {
    uint p = i * G_W + me;
    Carry carry = carryShuttle[(gr - 1) * WIDTH + ((p + WIDTH - gr / H) % WIDTH)];
    uint k = line + BIG_HEIGHT * p;
    u[i] = carryAndWeightFinal(wu[i], carry, A[p], k);
  }

  fft_WIDTH(lds, u, smallTrig);

  write(G_W, NW, u, io, 0);
}

KERNEL(256) transposeW(CP(T2) in, P(T2) out) {
  local T lds[4096];
  transpose(WIDTH, BIG_HEIGHT, lds, in, out);
}

KERNEL(256) transposeH(CP(T2) in, P(T2) out) {
  local T lds[4096];
  transpose(BIG_HEIGHT, WIDTH, lds, in, out);
}

// from transposed to sequential.
KERNEL(256) transposeOut(CP(Word2) in, P(Word2) out) {
  local Word2 lds[4096];
  transposeWords(WIDTH, BIG_HEIGHT, lds, in, out);
}

// from sequential to transposed.
KERNEL(256) transposeIn(CP(Word2) in, P(Word2) out) {
  local Word2 lds[4096];
  transposeWords(BIG_HEIGHT, WIDTH, lds, in, out);
}

KERNEL(SMALL_HEIGHT / 2 / 4) square(P(T2) io) {
  uint W = SMALL_HEIGHT;
  uint H = ND / W;

  uint me = get_local_id(0);
  uint line1 = get_group_id(0);
  uint line2 = (H - line1) % H;
  uint g1 = transPos(line1, MIDDLE, WIDTH);
  uint g2 = transPos(line2, MIDDLE, WIDTH);

  T2 base = slowTrig(me * H + line1, W * H);
  T2 step = slowTrig(1, 8);
  
  for (uint i = 0; i < 4; ++i, base = mul(base, step)) {
    if (i == 0 && line1 == 0 && me == 0) {
      io[0]     = shl(foo(conjugate(io[0])), 2);
      io[W / 2] = shl(sq(conjugate(io[W / 2])), 3);    
    } else {
      uint k = g1 * W + i * (W / 8) + me;
      uint v = g2 * W + (W - 1) + (line1 == 0) - i * (W / 8) - me;
      T2 a = io[k];
      T2 b = conjugate(io[v]);
      T2 t = swap(base);
      X2(a, b);
      b = mul(b, conjugate(t));
      X2(a, b);
      a = sq(a);
      b = sq(b);
      X2(a, b);
      b = mul(b, t);
      X2(a, b);
      io[k] = conjugate(a);
      io[v] = b;
    }
  }
}

KERNEL(SMALL_HEIGHT / 2) multiply(P(T2) io, CP(T2) in) {
  uint W = SMALL_HEIGHT;
  uint H = ND / W;
  
  uint line1 = get_group_id(0);
  uint me = get_local_id(0);

  if (line1 == 0 && me == 0) {
    io[0]     = shl(conjugate(foo2(io[0], in[0])), 2);
    io[W / 2] = shl(conjugate(mul(io[W / 2], in[W / 2])), 3);
    return;
  }

  uint line2 = (H - line1) % H;
  uint g1 = transPos(line1, MIDDLE, WIDTH);
  uint g2 = transPos(line2, MIDDLE, WIDTH);
  uint k = g1 * W + me;
  uint v = g2 * W + (W - 1) - me + (line1 == 0);
  T2 a = io[k];
  T2 b = conjugate(io[v]);
  T2 t = swap(slowTrig(me * H + line1, W * H));
  X2(a, b);
  b = mul(b, conjugate(t));
  X2(a, b);

  T2 c = in[k];
  T2 d = conjugate(in[v]);
  X2(c, d);
  d = mul(d, conjugate(t));
  X2(c, d);

  a = mul(a, c);
  b = mul(b, d);

  X2(a, b);
  b = mul(b, t);
  X2(a, b);

  io[k] = conjugate(a);
  io[v] = b;
}

#if 0
// Alternative form
KERNEL(SMALL_HEIGHT / 2 / 4) multiply(P(T2) io, CP(T2) in) {
  uint W = SMALL_HEIGHT;
  uint H = ND / W;

  uint me = get_local_id(0);
  uint line1 = get_group_id(0);
  uint line2 = (H - line1) % H;
  uint g1 = transPos(line1, MIDDLE, WIDTH);
  uint g2 = transPos(line2, MIDDLE, WIDTH);

  T2 base = slowTrig(me * H + line1, W * H);
  T2 step = slowTrig(1, 8);

  for (uint i = 0; i < 4; ++i, base = mul(base, step)) {
    if (i == 0 && line1 == 0 && me == 0) {
      io[0]     = shl(foo2(conjugate(io[0]), conjugate(in[0])), 2);
      io[W / 2] = shl(conjugate(mul(io[W / 2], in[W / 2])), 3);
    } else {
      uint k = g1 * W + i * (W / 8) + me;
      uint v = g2 * W + (W - 1) + (line1 == 0) - i * (W / 8) - me;
      T2 a = io[k];
      T2 b = conjugate(io[v]);
      T2 t = swap(base);
      X2(a, b);
      b = mul(b, conjugate(t));
      X2(a, b);

      T2 c = in[k];
      T2 d = conjugate(in[v]);
      X2(c, d);
      d = mul(d, conjugate(t));
      X2(c, d);

      a = mul(a, c);
      b = mul(b, d);

      X2(a, b);
      b = mul(b, t);
      X2(a, b);

      io[k] = conjugate(a);
      io[v] = b;
    }
  }
}
#endif

// tailFused below

void reverse(uint WG, local T *rawLds, T2 *u, bool bump) {
  local T2 *lds = (local T2 *)rawLds;
  uint me = get_local_id(0);
  uint revMe = WG - 1 - me + bump;
  
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
  for (int i = 0; i < NH/2; ++i) { u[i] = lds[i * WG + me]; }
}

void reverseLine(uint WG, local T *lds, T2 *u) {
  uint me = get_local_id(0);
  uint revMe = WG - 1 - me;
  for (int b = 0; b < 2; ++b) {
    bar();
    for (int i = 0; i < NH; ++i) { lds[i * WG + revMe] = ((T *) (u + ((NH - 1) - i)))[b]; }  
    bar();
    for (int i = 0; i < NH; ++i) { ((T *) (u + i))[b] = lds[i * WG + me]; }
  }
}

#ifdef ORIG_PAIRSQ

void pairSq(uint N, T2 *u, T2 *v, T2 base, bool special) {
  uint me = get_local_id(0);

  T2 step = slowTrig(1, NH);
  
  for (int i = 0; i < N; ++i, base = mul(base, step)) {
    T2 a = u[i];
    T2 b = conjugate(v[i]);
    T2 t = swap(base);    
    if (special && i == 0 && me == 0) {
      a = shl(foo(a), 2);
      b = shl(sq(b), 3);
    } else {
      X2(a, b);
      b = mul(b, conjugate(t));
      X2(a, b);
      a = sq(a);
      b = sq(b);
      X2(a, b);
      b = mul(b, t);
      X2(a, b);
    }
    u[i] = conjugate(a);
    v[i] = b;
  }
}

#else

// Better pairSq.  Reduces complex muls in base calculations and improves roundoff error

// This can be done with 31 float ops
#define onePairSq(a, b, t) { \
      b = conjugate(b); \
      X2(a, b); \
      b = mul_by_conjugate(b, t); \
      X2(a, b); \
      a = sq(a); \
      b = sq(b); \
      X2(a, b); \
      b = mul(b, t); \
      X2(a, b); \
      a = conjugate(a); \
}

void pairSq(uint N, T2 *u, T2 *v, T2 base, bool special) {
  uint me = get_local_id(0);

// Should assert N == NH/2 or N == NH

  T2 step = slowTrig(1, NH);

  for (int i = 0; i < NH / 4; ++i, base = mul(base, step)) {
    T2 a = u[i];
    T2 b = v[i];
    T2 t = swap(base);    
    if (special && i == 0 && me == 0) {
      b = conjugate(b);
      a = shl(foo(a), 2);
      b = shl(sq(b), 3);
      a = conjugate(a);
    } else {
      onePairSq(a, b, t);
    }
    u[i] = a;
    v[i] = b;

    if (N == NH) {
	a = u[i+NH/2];
	b = v[i+NH/2];
	t = swap(mul (base, U2(0, -1)));    
	onePairSq(a, b, t);
	u[i+NH/2] = a;
	v[i+NH/2] = b;
    }

    a = u[i+NH/4];
    b = v[i+NH/4];
    T2 new_base = mul_t8 (base);
    t = swap (new_base);
    onePairSq(a, b, t);
    u[i+NH/4] = a;
    v[i+NH/4] = b;

    if (N == NH) {
	a = u[i+3*NH/4];
	b = v[i+3*NH/4];
	t = swap(mul (new_base, U2(0, -1)));
	onePairSq(a, b, t);
	u[i+3*NH/4] = a;
	v[i+3*NH/4] = b;
    }
  }
}

#endif

#ifdef ORIG_PAIRMUL

void pairMul(uint N, T2 *u, T2 *v, T2 *p, T2 *q, T2 base, bool special) {
  uint me = get_local_id(0);

  T2 step = slowTrig(1, NH);
  
  for (int i = 0; i < N; ++i, base = mul(base, step)) {
    T2 a = u[i];
    T2 b = conjugate(v[i]);
    T2 c = p[i];
    T2 d = conjugate(q[i]);
    T2 t = swap(base);
    if (special && i == 0 && me == 0) {
      a = shl(foo2(a, c), 2);
      b = shl(mul(b, d), 3);
    } else {
      X2(a, b);
      b = mul(b, conjugate(t));
      X2(a, b);
      X2(c, d);
      d = mul(d, conjugate(t));
      X2(c, d);
      a = mul(a, c);
      b = mul(b, d);
      X2(a, b);
      b = mul(b, t);
      X2(a, b);
    }
    u[i] = conjugate(a);
    v[i] = b;
  }
}

#else

// Better pairMul.  Reduces complex muls in base calculations and improves roundoff error

#define onePairMul(a, b, c, d, t) { \
      b = conjugate(b); \
      X2(a, b); \
      b = mul_by_conjugate(b, t); \
      X2(a, b); \
      d = conjugate(d); \
      X2(c, d); \
      d = mul_by_conjugate(d, t); \
      X2(c, d); \
      a = mul(a, c); \
      b = mul(b, d); \
      X2(a, b); \
      b = mul(b, t); \
      X2(a, b); \
      a = conjugate(a); \
}

void pairMul(uint N, T2 *u, T2 *v, T2 *p, T2 *q, T2 base, bool special) {
  uint me = get_local_id(0);

  T2 step = slowTrig(1, NH);
  
  for (int i = 0; i < N/4; ++i, base = mul(base, step)) {
    T2 a = u[i];
    T2 b = v[i];
    T2 c = p[i];
    T2 d = q[i];
    T2 t = swap(base);
    if (special && i == 0 && me == 0) {
      b = conjugate(b);
      d = conjugate(d);
      a = shl(foo2(a, c), 2);
      b = shl(mul(b, d), 3);
      a = conjugate(a);
    } else {
      onePairMul(a, b, c, d, t);
    }
    u[i] = a;
    v[i] = b;

    if (N == NH) {
	a = u[i+NH/2];
	b = v[i+NH/2];
	c = p[i+NH/2];
	d = q[i+NH/2];
	t = swap(mul (base, U2(0, -1)));    
	onePairMul(a, b, c, d, t);
	u[i+NH/2] = a;
	v[i+NH/2] = b;
    }

    a = u[i+NH/4];
    b = v[i+NH/4];
    c = p[i+NH/4];
    d = q[i+NH/4];
    T2 new_base = mul_t8 (base);
    t = swap (new_base);
    onePairMul(a, b, c, d, t);
    u[i+NH/4] = a;
    v[i+NH/4] = b;

    if (N == NH) {
	a = u[i+3*NH/4];
	b = v[i+3*NH/4];
	c = p[i+3*NH/4];
	d = q[i+3*NH/4];
	t = swap(mul (new_base, U2(0, -1)));
	onePairMul(a, b, c, d, t);
	u[i+3*NH/4] = a;
	v[i+3*NH/4] = b;
    }
  }
}

#endif

// equivalent to: fftH, multiply, fftH.
KERNEL(G_H) tailFused(P(T2) io, Trig smallTrig) {
  local T2 rawLds[(SMALL_HEIGHT+1)/2];
  local T *lds = (local T *)rawLds;
  T2 u[NH], v[NH];

  uint W = SMALL_HEIGHT;
  uint H = ND / W;

  uint line1 = get_group_id(0);
  uint line2 = line1 ? H - line1 : (H / 2);
  uint g1 = transPos(line1, MIDDLE, WIDTH);
  uint g2 = transPos(line2, MIDDLE, WIDTH);
  
  read(G_H, NH, u, io, g1 * SMALL_HEIGHT);
  read(G_H, NH, v, io, g2 * SMALL_HEIGHT);
  fft_HEIGHT(lds, u, smallTrig);
  fft_HEIGHT(lds, v, smallTrig);

  uint me = get_local_id(0);
  if (line1 == 0) {
    // Line 0 is special: it pairs with itself, offseted by 1.
    reverse(G_H, lds, u + NH/2, true);
    pairSq(NH/2, u,   u + NH/2, slowTrig(me, W), true);
    reverse(G_H, lds, u + NH/2, true);

    // Line H/2 also pairs with itself (but without offset).
    reverse(G_H, lds, v + NH/2, false);
    pairSq(NH/2, v,   v + NH/2, slowTrig(1 + 2 * me, 2 * W), false);
    reverse(G_H, lds, v + NH/2, false);
  } else {    
    reverseLine(G_H, lds, v);
    pairSq(NH, u, v, slowTrig(line1 + me * H, ND), false);
    reverseLine(G_H, lds, v);
  }

  fft_HEIGHT(lds, v, smallTrig);
  write(G_H, NH, v, io, g2 * SMALL_HEIGHT);
  
  fft_HEIGHT(lds, u, smallTrig);
  write(G_H, NH, u, io, g1 * SMALL_HEIGHT);
}

// equivalent to: fftH(io), multiply(io, a - b), fftH(io)
KERNEL(G_H) tailFusedMulDelta(P(T2) io, CP(T2) a, CP(T2) b, Trig smallTrig) {
  local T2 rawLds[(SMALL_HEIGHT+1)/2];
  local T *lds = (local T *)rawLds;
  T2 u[NH], v[NH];
  T2 p[NH], q[NH];

  uint W = SMALL_HEIGHT;
  uint H = ND / W;

  uint line1 = get_group_id(0);
  uint line2 = line1 ? H - line1 : (H / 2);
  uint g1 = transPos(line1, MIDDLE, WIDTH);
  uint g2 = transPos(line2, MIDDLE, WIDTH);
  
  read(G_H, NH, u, io, g1 * SMALL_HEIGHT);
  read(G_H, NH, v, io, g2 * SMALL_HEIGHT);
  
  readDelta(G_H, NH, p, a, b, g1 * SMALL_HEIGHT);
  readDelta(G_H, NH, q, a, b, g2 * SMALL_HEIGHT);

  fft_HEIGHT(lds, u, smallTrig);
  fft_HEIGHT(lds, v, smallTrig);

  uint me = get_local_id(0);
  if (line1 == 0) {
    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, slowTrig(me, W), true);
    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);

    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, slowTrig(1 + 2 * me, 2 * W), false);
    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
  } else {    
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);
    pairMul(NH, u, v, p, q, slowTrig(line1 + me * H, W * H), false);
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);
  }

  fft_HEIGHT(lds, v, smallTrig);
  write(G_H, NH, v, io, g2 * SMALL_HEIGHT);
  
  fft_HEIGHT(lds, u, smallTrig);
  write(G_H, NH, u, io, g1 * SMALL_HEIGHT);
}

#define TEST_KERNEL	// Generate a small unused kernel so developers can look at how well individual macros assemble and optimize
#ifdef TEST_KERNEL
// Small test kernel so we can easily find code snipets to compare different implementations of macros
KERNEL(256) testKernel(P(T2) io) {
      fft4(io);
//    fft8(io);
//    fft10(io);
//    pairSq(NH, io, io+100, slowTrig(14, ND), false);
//    pairMul(NH, io, io+100, io+200, io+300, slowTrig(14, ND), false);
}
#endif


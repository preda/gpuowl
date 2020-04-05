// gpuOwl, an OpenCL Mersenne primality test.
// Copyright Mihai Preda and George Woltman.

/* List of user-serviceable -use flags and their effects

DEBUG  : enable asserts. Slow, but allows to verify that all asserts hold.
   
NO_ASM : request to not use any inline __asm()
NO_OMOD: do not use GCN output modifiers in __asm()

OUT_WG,OUT_SIZEX,OUT_SPACING <AMD default is 256,32,4> <nVidia default is 256,4,1 but needs testing>
IN_WG,IN_SIZEX,IN_SPACING <AMD default is 256,32,1>  <nVidia default is 256,4,1 but needs testing>

UNROLL_WIDTH <nVidia default>
NO_UNROLL_WIDTH <AMD default>

OLD_FFT8 <default>
NEWEST_FFT8
NEW_FFT8

OLD_FFT5
NEW_FFT5 <default>
NEWEST_FFT5

NEW_FFT10 <default>
OLD_FFT10

CARRY32 <AMD default for PRP when appropriate>
CARRY64 <nVidia default>, <AMD default for PM1 when appropriate>

CARRYM32
CARRYM64 <default>

ORIG_SLOWTRIG
NEW_SLOWTRIG <default>          // Our own sin/cos implementation
ROCM_SLOWTRIG                   // Use ROCm's private reduced-argument sin/cos

ROCM31    <AMD default>         // Enable workaround for ROCm 3.1 bug affecting kcos()
NO_ROCM31 <nVidia default>

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
PM1        set to indicate that a P-1 test will be done (as opposed to PRP)

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

#if __OPENCL_VERSION__ < 200
#pragma message "GpuOwl requires OpenCL 200, found " STR(__OPENCL_VERSION__)
// #error OpenCL >= 2.0 required
#endif

#pragma OPENCL FP_CONTRACT ON

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

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

#if CARRY32 && CARRY64
#error Conflict: both CARRY32 and CARRY64 requested
#endif

#if CARRYM32 && CARRYM64
#error Conflict: both CARRYM32 and CARRYM64 requested
#endif

#if !CARRY32 && !CARRY64
#if AMDGPU
#define CARRY32 1
#endif
#endif

#if !CARRYM32 && !CARRYM64
#define CARRYM64 1
#endif

#if !ROCM31 && !NO_ROCM31 && AMDGPU
#define ROCM31 1
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

#if !ORIG_SLOWTRIG && !NEW_SLOWTRIG
#define NEW_SLOWTRIG 1
#endif

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


// 5M timings for MiddleIn & tailFused, ROCm 2.10, RadeonVII, sclk4, mem 1200
// IN_WG=256, IN_SIZEX=4, IN_SPACING=1 (old WorkingIn4) : 177 + 164 (but may be best on nVidia)
// IN_WG=256, IN_SIZEX=8, IN_SPACING=1 (old WorkingIn3): 129 + 166 = 295
// IN_WG=256, IN_SIZEX=32, IN_SPACING=1 (old WorkingIn5): 107 + 171 = 278  <- best
// IN_WG=256, IN_SIZEX=8, IN_SPACING=2: 139 + 166 = 305
// IN_WG=256, IN_SIZEX=32, IN_SPACING=4: 121 + 161 = 282

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

#if !IN_SPACING
#if AMDGPU
#define IN_SPACING 1
#else
#define IN_SPACING 1
#endif
#endif

#if UNROLL_WIDTH
#define UNROLL_WIDTH_CONTROL
#else
#define UNROLL_WIDTH_CONTROL       __attribute__((opencl_unroll_hint(1)))
#endif

#if HAS_ASM && !NO_OMOD
// turn IEEE mode and denormals off so that mul:2 and div:2 work
#define ENABLE_MUL2() {\
    __asm("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 9, 1), 0");\
    __asm("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), 7");\
}
#else
#define ENABLE_MUL2()
#endif

typedef int i32;
typedef uint u32;
typedef long i64;
typedef ulong u64;
typedef double T;
typedef double2 T2;
typedef i32 Word;
typedef int2 Word2;

// Because the carry output of both carryA and carryM is consummed by carryB,
// we can use 32-bit in carry A/B/M only when carryA and carryM agree.
#if CARRY32 && CARRYM32
typedef i32 CarryABM;
#else
typedef i64 CarryABM;
#endif

T2 U2(T a, T b) { return (T2)(a, b); }

bool test(u32 bits, u32 pos) { return (bits >> pos) & 1; }

#define STEP (NWORDS - (EXP % NWORDS))
bool isBigWord(u32 extra) { return extra < NWORDS - STEP; }
u32 bitlen(bool b) { return EXP / NWORDS + b; }
u32 reduce(u32 extra) { return extra < NWORDS ? extra : (extra - NWORDS); }

T add1_m2(T x, T y) {
#if !NO_OMOD
  T tmp;
   __asm("v_add_f64 %0, %1, %2 mul:2" : "=v" (tmp) : "v" (x), "v" (y));
   return tmp;
#else
   return 2 * (x + y);
#endif
}

T sub1(T x, T y) {
#if !NO_OMOD
  T tmp;
  __asm("v_add_f64 %0, %1, -%2" : "=v" (tmp) : "v" (x), "v" (y));
  return tmp;
#else
  return x - y;
#endif  
}

T sub1_m2(T x, T y) {
#if !NO_OMOD
  T tmp;
  __asm("v_add_f64 %0, %1, -%2 mul:2" : "=v" (tmp) : "v" (x), "v" (y));
  return tmp;
#else
  return 2 * (x - y);
#endif
}

// x * y * 2
T mul1_m2(T x, T y) {
#if !NO_OMOD
  T tmp;
  __asm("v_mul_f64 %0, %1, %2 mul:2" : "=v" (tmp) : "v" (x), "v" (y));
  return tmp;
#else
  return x * y * 2;
#endif
}

T mad1(T x, T y, T z) { return x * y + z; }

T mad1_m2(T a, T b, T c) {
#if !NO_OMOD
  double out;
  __asm("v_fma_f64 %0, %1, %2, %3 mul:2" : "=v" (out) : "v" (a), "v" (b), "v" (c));
  return out;
#else
  return 2 * mad1(a, b, c);
#endif
}

T msb1_m2(T a, T b, T c) {
#if !NO_OMOD
  double out;
  __asm("v_fma_f64 %0, %1, %2, -%3 mul:2" : "=v" (out) : "v" (a), "v" (b), "v" (c));
  return out;
#else
  return 2 * mad1(a, b, -c);
#endif
}

T mad1_m4(T a, T b, T c) {
#if !NO_OMOD
  double out;
  __asm("v_fma_f64 %0, %1, %2, %3 mul:4" : "=v" (out) : "v" (a), "v" (b), "v" (c));
  return out;
#else
  return 4 * mad1(a, b, c);
#endif
}

T msb1_m4(T a, T b, T c) {
#if !NO_OMOD
  double out;
  __asm("v_fma_f64 %0, %1, %2, -%3 mul:4" : "=v" (out) : "v" (a), "v" (b), "v" (c));
  return out;
#else
  return 4 * mad1(a, b, -c);
#endif
}


// complex add * 2
T2 add_m2(T2 a, T2 b) { return U2(add1_m2(a.x, b.x), add1_m2(a.y, b.y)); }

// complex square
T2 sq(T2 a) { return U2(mad1(a.x, a.x, - a.y * a.y), mul1_m2(a.x, a.y)); }

// complex mul
T2 mul(T2 a, T2 b) { return U2(mad1(a.x, b.x, - a.y * b.y), mad1(a.x, b.y, a.y * b.x)); }

// complex mul * 2
T2 mul_m2(T2 a, T2 b) { return U2(msb1_m2(a.x, b.x, a.y * b.y), mad1_m2(a.x, b.y, a.y * b.x)); }

// complex mul * 4
T2 mul_m4(T2 a, T2 b) { return U2(msb1_m4(a.x, b.x, a.y * b.y), mad1_m4(a.x, b.y, a.y * b.x)); }

// complex fma
T2 mad_m1(T2 a, T2 b, T2 c) { return U2(mad1(a.x, b.x, mad1(a.y, -b.y, c.x)), mad1(a.x, b.y, mad1(a.y, b.x, c.y))); }

// complex fma * 2
T2 mad_m2(T2 a, T2 b, T2 c) { return U2(mad1_m2(a.x, b.x, mad1(a.y, -b.y, c.x)), mad1_m2(a.x, b.y, mad1(a.y, b.x, c.y))); }


T2 mul_t4(T2 a)  { return U2(a.y, -a.x); }                          // mul(a, U2( 0, -1)); }
T2 mul_t8(T2 a)  { return U2(a.y + a.x, a.y - a.x) * M_SQRT1_2; }   // mul(a, U2( 1, -1)) * (T)(M_SQRT1_2); }
T2 mul_3t8(T2 a) { return U2(a.x - a.y, a.x + a.y) * - M_SQRT1_2; } // mul(a, U2(-1, -1)) * (T)(M_SQRT1_2); }

T2 swap(T2 a) { return U2(a.y, a.x); }
T2 conjugate(T2 a) { return U2(a.x, -a.y); }

void bar() { barrier(0); }

T2 weight(Word2 a, T2 w) { return U2(a.x, a.y) * w; }

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

double optionalDouble(double iw) {
  // In a straightforward implementation, inverse weights are between 0.5 and 1.0.  We use inverse weights between 1.0 and 2.0
  // because it allows us to implement this routine with a single OR instruction on the exponent.   The original implementation
  // where this routine took as input values from 0.25 to 1.0 required both an AND and an OR instruction on the exponent.
  // return iw <= 1.0 ? iw * 2 : iw;
  assert(iw > 0.5 && iw < 2);
  uint2 u = as_uint2(iw);
  // u.y |= 0x00100000;
  u.y = bfi(u.y, 0xffefffff, 0x00100000);
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
u32 ulowBits(u32 u, u32 bits) { u32 tmp; __asm("v_bfe_u32 %0, %1, 0, %2" : "=v" (tmp) : "v" (u), "v" (bits)); return tmp; }
#else
i32  lowBits(i32 u, u32 bits) { return ((u << (32 - bits)) >> (32 - bits)); }
u32 ulowBits(u32 u, u32 bits) { return ((u << (32 - bits)) >> (32 - bits)); }
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

uint OVERLOAD roundoff(double x) { return rint(ldexp(fabs((float) (x - rint(x))), 32)); }
uint OVERLOAD roundoff(double2 x) { return max(roundoff(x.x), roundoff(x.y)); }

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
  *outCarry = (x - w) >> nBits;
  return w;
}

Word OVERLOAD carryStep(i32 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);
  *outCarry = (x - w) >> nBits;
  CARRY32_CHECK(*outCarry);
  return w;
}

Word OVERLOAD carryStep(i64 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);
#if 0
  i32 out;
  __asm("v_alignbit_b32 %0, %1, %2, %3" : "=v"(out) : "v"(as_int2(x).y), "v"(as_int2(x).x), "v"(nBits));
  *outCarry = out + (w < 0);
#else
  *outCarry = ((i32) (x >> nBits)) + (w < 0);
#endif
  CARRY32_CHECK(*outCarry);
  return w;
}

#if CARRY32
typedef i32 CFcarry;
#else
typedef i64 CFcarry;
#endif

#if CARRYM32
typedef i32 CFMcarry;
#else
typedef i64 CFMcarry;
#endif

u32 bound(i64 carry) { return min(abs(carry), 0xfffffffful); }

//{{ carries
Word2 OVERLOAD unweightAndCarry(T2 u, iCARRY *outCarry, bool b1, bool b2, iCARRY inCarry, u32* carryMax) {
  iCARRY midCarry;
  Word a = carryStep(doubleToLong(u.x, inCarry), &midCarry, b1);
  Word b = carryStep(doubleToLong(u.y, midCarry), outCarry, b2);
#if ROUNDOFF
  *carryMax = max(*carryMax, max(bound(midCarry), bound(*outCarry)));
#endif
  return (Word2) (a, b);
}

Word2 OVERLOAD unweightAndCarryMul(T2 u, iCARRY *outCarry, bool b1, bool b2, iCARRY inCarry, u32* carryMax) {
  iCARRY midCarry;
  Word a = carryStep(3 * doubleToLong(u.x, (iCARRY) 0) + inCarry, &midCarry, b1);
  Word b = carryStep(3 * doubleToLong(u.y, (iCARRY) 0) + midCarry, outCarry, b2);
#if ROUNDOFF
  *carryMax = max(*carryMax, max(bound(midCarry), bound(*outCarry)));
#endif
  return (Word2) (a, b);
}

T2 OVERLOAD carryAndWeightFinal(Word2 u, iCARRY inCarry, T2 w, bool b1) {
  i32 tmpCarry;
  u.x = carryStep(u.x + inCarry, &tmpCarry, b1);
  u.y += tmpCarry;
  return U2(u.x, u.y) * w;
}
//}}

//== carries CARRY=32
//== carries CARRY=64

// Carry propagation from word and carry.
Word2 carryWord(Word2 a, CarryABM* carry, bool b1, bool b2) {
  a.x = carryStep(a.x + *carry, carry, b1);
  a.y = carryStep(a.y + *carry, carry, b2);
  return a;
}

// Propagate carry this many pairs of words.
#define CARRY_LEN 16

//
// More miscellaneous macros
//

T2 addsub(T2 a) { return U2(a.x + a.y, a.x - a.y); }
T2 addsub_m2(T2 a) { return U2(add1_m2(a.x, a.y), sub1_m2(a.x, a.y)); }

// computes 2*(a.x*b.x+a.y*b.y) + i*2*(a.x*b.y+a.y*b.x)
T2 foo2(T2 a, T2 b) {
  a = addsub(a);
  b = addsub(b);
  return addsub(U2(a.x * b.x, a.y * b.y));
}

T2 foo2_m2(T2 a, T2 b) {
  a = addsub(a);
  b = addsub(b);
  return addsub_m2(U2(a.x * b.x, a.y * b.y));
}

// computes 2*[x^2+y^2 + i*(2*x*y)]. Needs a name.
T2 foo(T2 a) { return foo2(a, a); }
T2 foo_m2(T2 a) { return foo2_m2(a, a); }

// Same as X2(a, b), b = mul_t4(b)
#define X2_mul_t4(a, b) { T2 t = a; a = t + b; t.x = sub1(b.x, t.x); b.x = sub1(t.y, b.y); b.y = t.x; }

#define X2(a, b) { T2 t = a; a = t + b; b = t - b; }
// { T2 t = a; a = t + b; b.x = sub1(t.x, b.x); b.y = sub1(t.y, b.y); }

// Same as X2(a, conjugate(b))
#define X2conjb(a, b) { T2 t = a; a.x = a.x + b.x; a.y = a.y - b.y; b.x = t.x - b.x; b.y = t.y + b.y; }

// Same as X2(a, b), a = conjugate(a)
#define X2conja(a, b) { T2 t = a; a.x = a.x + b.x; a.y = -a.y - b.y; b = t - b; }

#define SWAP(a, b) { T2 t = a; a = b; b = t; }

T2 fmaT2(T a, T2 b, T2 c) { return U2(a * b.x + c.x, a * b.y + c.y); }

#define fma_addsub(a, b, sin, c, d) { d = sin * d; a = d + c; b = -d + c; }
// #define fma_addsub(a, b, sin, c, d) { a = fmaT2(sin, d, c); b = fmaT2(sin, -d, c); }

// a * conjugate(b)
// saves one negation
T2 mul_by_conjugate(T2 a, T2 b) { return U2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y); }

// Combined complex mul and mul by conjugate.  Saves 4 multiplies compared to two complex mul calls. 
void mul_and_mul_by_conjugate(T2 *res1, T2 *res2, T2 a, T2 b) {
	T axbx = a.x * b.x; T axby = a.x * b.y; T aybx = a.y * b.x; T ayby = a.y * b.y;
	res1->x = axbx - ayby; res1->y = axby + aybx;		// Complex mul
	res2->x = axbx + ayby; res2->y = aybx - axby;		// Complex mul by conjugate
}

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

#if !OLD_FFT8 && !NEWEST_FFT8 && !NEW_FFT8
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
				 a.x = fma(b.x, M_SQRT1_2, t.x); a.y = fma(b.y, M_SQRT1_2, t.y); \
				 b.x = fma(b.x, -M_SQRT1_2, t.x); b.y = fma(b.y, -M_SQRT1_2, t.y); }

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


// FFT routines to implement the middle step

void fft3(T2 *u) {
  const double SQRT3_2 = 0x1.bb67ae8584caap-1; // sin(tau/3), sqrt(3)/2, 0.86602540378443859659;
  
  X2(u[1], u[2]);
  T2 u0 = u[0];
  u[0] += u[1];
  u[1] = u0 - u[1] / 2;
  u[2] = mul_t4(u[2] * SQRT3_2);
  X2(u[1], u[2]);
}

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


void fft6(T2 *u) {
  const double SQRT3_2 = 0x1.bb67ae8584caap-1; // sin(tau/3), sqrt(3)/2, 0.86602540378443859659;

  for (i32 i = 0; i < 3; ++i) { X2(u[i], u[i + 3]); }
  
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

// See prime95's gwnum/zr8.mac file for more detailed explanation of the formulas below
// R1 = ((r1+r5)+(r3+r7)) +((r2+r6)+(r4+r8))
// R5 = ((r1+r5)+(r3+r7)) -((r2+r6)+(r4+r8))
// R3 = ((r1+r5)-(r3+r7))                                    +((i2+i6)-(i4+i8))
// R7 = ((r1+r5)-(r3+r7))                                    -((i2+i6)-(i4+i8))
// R2 = ((r1-r5)      +.707((r2-r6)-(r4-r8)))  + ((i3-i7)+.707((i2-i6)+(i4-i8)))
// R8 = ((r1-r5)      +.707((r2-r6)-(r4-r8)))  - ((i3-i7)+.707((i2-i6)+(i4-i8)))
// R4 = ((r1-r5)      -.707((r2-r6)-(r4-r8)))  - ((i3-i7)-.707((i2-i6)+(i4-i8)))
// R6 = ((r1-r5)      -.707((r2-r6)-(r4-r8)))  + ((i3-i7)-.707((i2-i6)+(i4-i8)))

// I1 = ((i1+i5)+(i3+i7)) +((i2+i6)+(i4+i8))
// I5 = ((i1+i5)+(i3+i7)) -((i2+i6)+(i4+i8))
// I3 = ((i1+i5)-(i3+i7))                                    -((r2+r6)-(r4+r8))
// I7 = ((i1+i5)-(i3+i7))                                    +((r2+r6)-(r4+r8))
// I2 = ((i1-i5)      +.707((i2-i6)-(i4-i8)))  - ((r3-r7)+.707((r2-r6)+(r4-r8)))
// I8 = ((i1-i5)      +.707((i2-i6)-(i4-i8)))  + ((r3-r7)+.707((r2-r6)+(r4-r8)))
// I4 = ((i1-i5)      -.707((i2-i6)-(i4-i8)))  + ((r3-r7)-.707((r2-r6)+(r4-r8)))
// I6 = ((i1-i5)      -.707((i2-i6)-(i4-i8)))  - ((r3-r7)-.707((r2-r6)+(r4-r8)))

void fft8mid(T2 *u) {
  X2(u[0], u[4]);					// (r1+ i1+), (r1-  i1-)
  X2_mul_t4(u[1], u[5]);				// (r2+ i2+), (i2- -r2-)
  X2_mul_t4(u[2], u[6]);				// (r3+ i3+), (i3- -r3-)
  X2_mul_t4(u[3], u[7]);				// (r4+ i4+), (i4- -r4-)

  X2(u[0], u[2]);					// (r1++  i1++), ( r1+-  i1+-)
  X2_mul_t4(u[1], u[3]);				// (r2++  i2++), ( i2+- -r2+-)
  X2_mul_t4(u[5], u[7]);				// (i2-+ -r2-+), (-r2-- -i2--)

  T2 tmp28a = fmaT2(-M_SQRT1_2, u[7], u[4]);
  T2 tmp46a = fmaT2(M_SQRT1_2, u[7], u[4]);
  T2 tmp28b = fmaT2(M_SQRT1_2, u[5], u[6]);
  T2 tmp46b = fmaT2(-M_SQRT1_2, u[5], u[6]);

  u[4] = u[0] - u[1];
  u[0] = u[0] + u[1];
  u[6] = u[2] - u[3];
  u[2] = u[2] + u[3];

  u[1] = tmp28a + tmp28b;
  u[7] = tmp28a - tmp28b;
  u[3] = tmp46a - tmp46b;
  u[5] = tmp46a + tmp46b;
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

#if !NEW_FFT10 && !OLD_FFT10
#define NEW_FFT10 1
#endif

// Using rocm 2.9, testKernel shows this macro generates a non-optimal 108 f64 ops (40 FMA), 64 vgprs (using NEW_FFT5).
#if OLD_FFT10
void fft10(T2 *u) {
  const double COS1 =  0x1.9e3779b97f4a8p-1; // cos(tau/10), 0.80901699437494745126
  const double SIN1 = -0x1.2cf2304755a5ep-1; // sin(tau/10), 0.58778525229247313710
  const double COS2 =  0x1.3c6ef372fe95p-2;  // cos(tau/5),  0.30901699437494745126
  const double SIN2 = -0x1.e6f0e134454ffp-1; // sin(tau/5),  0.95105651629515353118

  for (i32 i = 0; i < 5; ++i) { X2(u[i], u[i + 5]); }
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

// Using rocm 2.9, testKernel shows this macro generates 92 f64 ops (24 FMA) or 84 f64 ops (40 FMA), 80 vgprs.
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
  const double SIN1 = 0x1.e6f0e134454ffp-1;		// sin(tau/5), 0.95105651629515353118
  const double SIN2_SIN1 = 0.618033988749894848;	// sin(2*tau/5) / sin(tau/5) = .588/.951, 0.618033988749894848
  const double COS1 = 0.309016994374947424;		// cos(tau/5), 0.309016994374947424
  const double COS2 = 0.809016994374947424;		// -cos(2*tau/5), 0.809016994374947424

  X2(u[0], u[5]);					// (r1+ i1+),  (r1-  i1-)
  X2_mul_t4(u[1], u[6]);				// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[4], u[9]);				// (r5+ i5+),  (i5- -r5-)
  X2_mul_t4(u[2], u[7]);				// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(u[3], u[8]);				// (r4+ i4+),  (i4- -r4-)

  X2_mul_t4(u[1], u[4]);				// (r2++  i2++),  (i2+- -r2+-)
  X2_mul_t4(u[6], u[9]);				// (i2-+ -r2-+), (-r2-- -i2--)
  X2_mul_t4(u[2], u[3]);				// (r3++  i3++),  (i3+- -r3+-)
  X2_mul_t4(u[7], u[8]);				// (i3-+ -r3-+), (-r3-- -i3--)

  T2 tmp39a = fmaT2(COS1, u[1], u[0]);
  T2 tmp57a = fmaT2(-COS2, u[1], u[0]);
  u[0] = u[0] + u[1];
  T2 tmp210a = fmaT2(-COS2, u[9], u[5]);
  T2 tmp48a = fmaT2(COS1, u[9], u[5]);
  u[5] = u[5] + u[9];

  T2 tmp39b = fmaT2(SIN2_SIN1, u[3], u[4]);		// (i2+- +.588/.951*i3+-, -r2+- -.588/.951*r3+-)
  T2 tmp57b = fmaT2(SIN2_SIN1, u[4], -u[3]);		// (.588/.951*i2+- -i3+-, -.588/.951*r2+- +r3+-)
  T2 tmp210b = fmaT2(SIN2_SIN1, u[6], u[7]);		// (.588/.951*i2-+ +i3-+, -.588/.951*r2-+ -r3-+)
  T2 tmp48b = fmaT2(-SIN2_SIN1, u[7], u[6]);		// (i2-+ -.588/.951*i3-+, -r2-+ +.588/.951*r3-+)

  tmp39a = fmaT2(-COS2, u[2], tmp39a);
  tmp57a = fmaT2(COS1, u[2], tmp57a);
  u[0] = u[0] + u[2];
  tmp210a = fmaT2(-COS1, u[8], tmp210a);
  tmp48a = fmaT2(COS2, u[8], tmp48a);
  u[5] = u[5] - u[8];

  fma_addsub(u[2], u[8], SIN1, tmp39a, tmp39b);
  fma_addsub(u[4], u[6], SIN1, tmp57a, tmp57b);
  fma_addsub(u[1], u[9], SIN1, tmp210a, tmp210b);
  fma_addsub(u[3], u[7], SIN1, tmp48a, tmp48b);
}
#else
#error None of OLD_FFT10, NEW_FFT10 defined
#endif


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

void shufl(u32 WG, local T2 *lds2, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  u32 m = me / f;
    
  local T* lds = (local T*) lds2;

  for (u32 i = 0; i < n; ++i) { lds[(m + i * WG / f) / n * f + m % n * WG + me % f] = u[i].x; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].x = lds[i * WG + me]; }
  bar();
  for (u32 i = 0; i < n; ++i) { lds[(m + i * WG / f) / n * f + m % n * WG + me % f] = u[i].y; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].y = lds[i * WG + me]; }
}

void shufl2(u32 WG, local T2 *lds2, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  local T* lds = (local T*) lds2;

  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~(f-1)) * n + me % f] = u[i].x; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].x = lds[i * WG + me]; }
  bar();
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~(f-1)) * n + me % f] = u[i].y; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].y = lds[i * WG + me]; }
}

void tabMul(u32 WG, const global T2 *trig, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  for (i32 i = 1; i < n; ++i) { u[i] = mul(u[i], trig[me / f + i * (WG / f)]); }
  // for (i32 i = 1; i < n; ++i) { u[i] = mul(u[i], trig[me / f * f + i * WG]); }
}

void shuflAndMul(u32 WG, local T2 *lds, const global T2 *trig, T2 *u, u32 n, u32 f) {
  shufl(WG, lds, u, n, f);
  tabMul(WG, trig, u, n, f);
}

void shuflAndMul2(u32 WG, local T2 *lds, const global T2 *trig, T2 *u, u32 n, u32 f) {
  tabMul(WG, trig, u, n, f);
  shufl2(WG, lds, u, n, f);
}

// 64x4
void fft256w(local T2 *lds, T2 *u, const global T2 *trig) {
  UNROLL_WIDTH_CONTROL
  for (i32 s = 4; s >= 0; s -= 2) {
    fft4(u);
    shuflAndMul(64, lds, trig, u, 4, 1 << s);
  }
  fft4(u);
}

void fft256h(local T2 *lds, T2 *u, const global T2 *trig) {
  /*
  UNROLL_WIDTH_CONTROL
  for (i32 s = 0; s <= 4; s += 2) {
    fft4(u);
    shuflAndMul2(64, lds, trig, u, 4, 1 << s);
  }
  fft4(u);
  */

  fft4(u);
  shuflAndMul2(64, lds, trig, u, 4, 1);
  fft4(u);
  shuflAndMul2(64, lds, trig, u, 4, 4);
  fft4(u);
  shuflAndMul2(64, lds, trig, u, 4, 16);
  fft4(u);
}

// 64x8
void fft512w(local T2 *lds, T2 *u, const global T2 *trig) {
  UNROLL_WIDTH_CONTROL
  for (i32 s = 3; s >= 0; s -= 3) {
    fft8(u);
    shuflAndMul(64, lds, trig, u, 8, 1 << s);
  }
  fft8(u);
}

void fft512h(local T2 *lds, T2 *u, const global T2 *trig) {
  fft8(u);
  shuflAndMul(64, lds, trig, u, 8, 8);
  fft8(u);
  shuflAndMul(64, lds, trig, u, 8, 1);
  fft8(u);
}

// 256x4
void fft1Kw(local T2 *lds, T2 *u, const global T2 *trig) {
  UNROLL_WIDTH_CONTROL
  for (i32 s = 0; s <= 6; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul2(256, lds, trig, u, 4, 1 << s);
  }
  fft4(u);
}
void fft1Kh(local T2 *lds, T2 *u, const global T2 *trig) {
  fft4(u);
  shuflAndMul(256, lds, trig, u, 4, 64);
  fft4(u);
  bar();
  shuflAndMul(256, lds, trig, u, 4, 16);
  fft4(u);
  bar();
  shuflAndMul(256, lds, trig, u, 4, 4);
  fft4(u);
  bar();
  shuflAndMul(256, lds, trig, u, 4, 1);
  fft4(u);
}

// 512x8
void fft4Kw(local T2 *lds, T2 *u, const global T2 *trig) {
  UNROLL_WIDTH_CONTROL
  for (i32 s = 6; s >= 0; s -= 3) {
    if (s != 6) { bar(); }
    fft8(u);
    shuflAndMul(512, lds, trig, u, 8, 1 << s);
  }
  fft8(u);
}
void fft4Kh(local T2 *lds, T2 *u, const global T2 *trig) {
  fft8(u);
  shuflAndMul(512, lds, trig, u, 8, 64);
  fft8(u);
  bar();
  shuflAndMul(512, lds, trig, u, 8, 8);
  fft8(u);
  bar();
  shuflAndMul(512, lds, trig, u, 8, 1);
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

double2 openclSlowTrig(i32 k, i32 n) {
  double c;
  double s = sincos(M_PI / n * k, &c);
  return U2(c, -s);
}

#if ROCM_SLOWTRIG
// ROCm OCML sin/cos which expect reduced argument.
// These are faster than openclSlowTrig() because they skip the argument reduction.
struct scret { double s; double c; };
extern struct scret __ocmlpriv_sincosred2_f64(double x, double y);
extern struct scret __ocmlpriv_sincosred_f64(double x);
double2 ocmlCosSin(double x) {
  // This has very similar accuracy and performance to our kCosSin(), and is affected by the same "mistify" bug.
  // struct scret r = __ocmlpriv_sincosred2_f64(x, 0);
  struct scret r = __ocmlpriv_sincosred_f64(x);
  return U2(r.c, r.s);
}
#endif

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
double ksin(double x) {
  // Coefficients from http://www.netlib.org/fdlibm/k_sin.c
  // Excellent accuracy in [-pi/4, pi/4]
  const double
  S1 = -0x1.5555555555555p-3,  // -1.66666666666666657415e-01 bfc55555'55555555
  S2 = +0x1.1111111110bb3p-7,  // +8.33333333333094970763e-03 3f811111'11110bb3
  S3 = -0x1.a01a019e83e5cp-13, // -1.98412698367611268872e-04 bf2a01a0'19e83e5c
  S4 = +0x1.71de3796cde01p-19, // +2.75573161037288024585e-06 3ec71de3'796cde01
  S5 = -0x1.ae600b42fdfa7p-26, // -2.50511320680216983368e-08 be5ae600'b42fdfa7
  S6 = +0x1.5e0b2f9a43bb8p-33; // +1.59181443044859141215e-10 3de5e0b2'f9a43bb8

  double z = x * x;
  return (((((S6 * z + S5) * z + S4) * z + S3) * z + S2) * z + S1) * z * x + x;
}

double kcos(double x) {
  // Coefficients from http://www.netlib.org/fdlibm/k_cos.c
  const double 
  C1  =  4.16666666666666019037e-02, /* 0x3FA55555, 0x5555554C */
  C2  = -1.38888888888741095749e-03, /* 0xBF56C16C, 0x16C15177 */
  C3  =  2.48015872894767294178e-05, /* 0x3EFA01A0, 0x19CB1590 */
  C4  = -2.75573143513906633035e-07, /* 0xBE927E4F, 0x809C52AD */
  C5  =  2.08757232129817482790e-09, /* 0x3E21EE9E, 0xBDB4B1C4 */
  C6  = -1.13596475577881948265e-11; /* 0xBDA8FAE9, 0xBE8838D4 */

  double z = x * x;
  double r = (((((C6 * z + C5) * z + C4) * z + C3) * z + C2) * z + C1) * z - 0.5;

#if ROCM31
  // The condition below is never hit,
  // it is here just to workaround a ROCm 3.1 maddening codegen bug.
  if (as_int2(x).y == -1) { return x; }  
#endif

#if HAS_ASM
  // the raw v_fma_f64 below seems to improve VGPR allocation on ROCm 3.3.0
  double out;
  __asm("v_fma_f64 %0, %1, %2, 1.0" : "=v"(out) : "v"(r), "v"(z));
  return out;
#else
  return r * z + 1;
#endif
}

double2 reducedCosSin(double x) {
  assert(x <= M_PI / 4);
#if ROCM_SLOWTRIG
  return ocmlCosSin(x);
#else
  return U2(kcos(x), ksin(x));
#endif
}

// Returns e^(-i * pi * k/n)
double2 slowTrig(i32 k, i32 n, i32 kBound) {
  assert(n % 2 == 0);
  assert(k >= 0);
  assert(k <= kBound);          // kBound actually bounds k
  assert(kBound <= 2 * n);      // angle <= 2*pi

#if ORIG_SLOWTRIG
  return openclSlowTrig(k, n);
#else
  if (kBound * 4 <= n) {        // angle <= pi/4
    double x = M_PI / n * k;
    double2 r = reducedCosSin(x);
    return U2(r.x, -r.y);
  } else if (kBound * 2 <= n) { // angle <= pi/2
    bool flip = kBound * 4 > n && k * 4 > n;
    if (flip) { k = n / 2 - k; }
    double x = M_PI / n * k;
    double2 r = reducedCosSin(x);
    if (flip) { r = swap(r); }
    return U2(r.x, -r.y);
    // return flip ? U2(r.y, - r.x) : U2(r.x, - r.y);
  } else {
    return openclSlowTrig(k, n);
  }
#endif
}

// transpose LDS 64 x 64.
void transposeLDS(local T *lds, T2 *u) {
  u32 me = get_local_id(0);
  for (i32 b = 0; b < 2; ++b) {
    if (b) { bar(); }
    for (i32 i = 0; i < 16; ++i) {
      u32 l = i * 4 + me / 64;
      lds[l * 64 + (me + l) % 64 ] = ((T *)(u + i))[b];
    }
    bar();
    for (i32 i = 0; i < 16; ++i) {
      u32 c = i * 4 + me / 64;
      u32 l = me % 64;
      ((T *)(u + i))[b] = lds[l * 64 + (c + l) % 64];
    }
  }
}


void transposeWords(u32 W, u32 H, local Word2 *lds, const Word2 *in, Word2 *out) {
  u32 GPW = W / 64, GPH = H / 64;

  u32 g = get_group_id(0);
  u32 gy = g % GPH;
  u32 gx = g / GPH;
  gx = (gy + gx) % GPW;

  in   += gy * 64 * W + gx * 64;
  out  += gy * 64     + gx * 64 * H;
  
  u32 me = get_local_id(0);
  u32 mx = me % 64;
  u32 my = me / 64;
  
  Word2 u[16];

  for (i32 i = 0; i < 16; ++i) { u[i] = in[(4 * i + my) * W + mx]; }

  for (i32 i = 0; i < 16; ++i) {
    u32 l = i * 4 + me / 64;
    lds[l * 64 + (me + l) % 64 ] = u[i];
  }
  bar();
  for (i32 i = 0; i < 16; ++i) {
    u32 c = i * 4 + me / 64;
    u32 l = me % 64;
    u[i] = lds[l * 64 + (c + l) % 64];
  }

  for (i32 i = 0; i < 16; ++i) {
    out[(4 * i + my) * H + mx] = u[i];
  }
}

#define P(x) global x * restrict
#define CP(x) const P(x)
typedef CP(T2) Trig;

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

// Read 64 Word2 starting at position 'startDword'.
KERNEL(64) readResidue(P(Word2) out, CP(Word2) in, u32 startDword) {
  u32 me = get_local_id(0);
  u32 k = (startDword + me) % ND;
  u32 y = k % BIG_HEIGHT;
  u32 x = k / BIG_HEIGHT;
  out[me] = in[WIDTH * y + x];
}

u32 transPos(u32 k, u32 width, u32 height) { return k / height + k % height * width; }

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
void readTailFusedLine(CP(T2) in, T2 *u, u32 line, u32 memline) {
  // We go to some length here to avoid dividing by MIDDLE in address calculations.
  // The transPos converted logical line number into physical memory line numbers
  // using this formula:  memline = line / WIDTH + line % WIDTH * MIDDLE.
  // We can compute the 0..9 component of address calculations as line / WIDTH,
  // and the 0,10,20,30,..310 component as (line % WIDTH) % 32 = (line % 32),
  // and the multiple of 320 component as (line % WIDTH) / 32

  u32 me = get_local_id(0);
  u32 WG = IN_WG * IN_SPACING;
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

  readTailFusedLine(in, u, g, transPos(g, MIDDLE, WIDTH));
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

// fftPremul: weight words with "A" (for IBDWT) followed by FFT.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, CP(T2) A, Trig smallTrig) {
  local T2 lds[WIDTH / 2];

  T2 u[NW];
  u32 g = get_group_id(0);

  u32 step = WIDTH * g;
  A   += step;
  in  += step;
  out += step;

  u32 me = get_local_id(0);

  for (i32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // u32 hk = g + BIG_HEIGHT * p;
    u[i] = weight(in[p], A[p]);
  }
  ENABLE_MUL2();

  fft_WIDTH(lds, u, smallTrig);
  
  write(G_W, NW, u, out, 0);
}

void fft_MIDDLE(T2 *u) {
#if MIDDLE == 1
  // Do nothing
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
  fft8mid(u);
#elif MIDDLE == 9
  fft9(u);
#elif MIDDLE == 10
  fft10(u);
#elif MIDDLE == 11
  fft11(u);
#elif MIDDLE == 12
  fft12(u);
#else
#error
#endif
}

// Apply the twiddles needed after fft_MIDDLE and before fft_HEIGHT in forward FFT.
// Also used after fft_HEIGHT and before fft_MIDDLE in inverse FFT.
void middleMul(T2 *u, u32 s) {
  assert(s < SMALL_HEIGHT);
  if (MIDDLE == 1) { return; }
  
  T2 step = slowTrig(s, BIG_HEIGHT / 2, BIG_HEIGHT / MIDDLE);
  u[1] = mul(u[1], step);
  
  T2 step2 = sq(step);
  u[2] = mul(u[2], step2);
  if (MIDDLE == 3) { return; }

  u[3] = mul(u[3], mul(step2, step));
  if (MIDDLE == 4) { return; }
  
  T2 base = sq(step2);
  for (i32 i = 4; i < MIDDLE; ++i) {
    u[i] = mul(u[i], base);
    base = mul(base, step);
  }
}

// Apply the twiddles needed after fft_WIDTH and before fft_MIDDLE in forward FFT.
// Also used after fft_MIDDLE and before fft_WIDTH in inverse FFT.
void middleMul2(T2 *u, u32 g, u32 me, double factor) {
  assert(g < WIDTH);
  assert(me < SMALL_HEIGHT);
  T2 base = slowTrig(g * me,           BIG_HEIGHT * WIDTH / 2, BIG_HEIGHT * WIDTH / MIDDLE) * factor;
  T2 step = slowTrig(g * SMALL_HEIGHT, BIG_HEIGHT * WIDTH / 2, BIG_HEIGHT * WIDTH / MIDDLE);
  for (i32 i = 0; i < MIDDLE; ++i) {
    u[i] = mul(u[i], base);
    base = mul(base, step);
  }
}

// Do a partial transpose during fftMiddleIn/Out
// The AMD OpenCL optimization guide indicates that reading/writing T values will be more efficient
// than reading/writing T2 values.  This routine lets us try both versions.

void middleShuffle(local T *lds, T2 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  local T *p = lds + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
  if (MIDDLE <= 8) {
    for (int i = 0; i < MIDDLE; ++i) { p[i * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].x = lds[me + workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p[i * workgroupSize] = u[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].y = lds[me + workgroupSize * i]; }
  } else {
    for (int i = 0; i < MIDDLE/2; ++i) { p[i * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE/2; ++i) { u[i].x = lds[me + workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE/2; ++i) { p[i * workgroupSize] = u[i].y; }
    bar();
    for (int i = 0; i < MIDDLE/2; ++i) { u[i].y = lds[me + workgroupSize * i]; }
    bar();
    for (int i = MIDDLE/2; i < MIDDLE; ++i) { p[(i - MIDDLE/2) * workgroupSize] = u[i].x; }
    bar();
    for (int i = MIDDLE/2; i < MIDDLE; ++i) { u[i].x = lds[me + workgroupSize * (i - MIDDLE/2)]; }
    bar();
    for (int i = MIDDLE/2; i < MIDDLE; ++i) { p[(i - MIDDLE/2) * workgroupSize] = u[i].y; }
    bar();
    for (int i = MIDDLE/2; i < MIDDLE; ++i) { u[i].y = lds[me + workgroupSize * (i - MIDDLE/2)]; }
  }
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

  ENABLE_MUL2();

  middleMul2(u, startx + mx, starty + my, 1);

  fft_MIDDLE(u);

  middleMul(u, starty + my);
  local T lds[IN_WG * (MIDDLE <= 8 ? MIDDLE : ((MIDDLE + 1) / 2))];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);

  out += gx * (MIDDLE * SMALL_HEIGHT * IN_SIZEX) + (gy / IN_SPACING) * (MIDDLE * IN_WG * IN_SPACING) + (gy % IN_SPACING) * SIZEY;
  out += (me / SIZEY) * (IN_SPACING * SIZEY) + (me % SIZEY);

  for (i32 i = 0; i < MIDDLE; ++i) { out[i * (IN_WG * IN_SPACING)] = u[i]; }
}

KERNEL(OUT_WG) fftMiddleOut(P(T2) out, P(T2) in) {
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

  middleMul(u, startx + mx);

  fft_MIDDLE(u);

  middleMul2(u, starty + my, startx + mx, 1.0 / (4 * (4 * NWORDS)));	// Compensate for weight and invweight being doubled
  local T lds[OUT_WG * (MIDDLE <= 8 ? MIDDLE : ((MIDDLE + 1) / 2))];
  middleShuffle(lds, u, OUT_WG, OUT_SIZEX);

  out += gx * (MIDDLE * WIDTH * OUT_SIZEX);
  out += (gy / OUT_SPACING) * (MIDDLE * (OUT_WG * OUT_SPACING));
  out += (gy % OUT_SPACING) * SIZEY;
  out += (me / SIZEY) * (OUT_SPACING * SIZEY);
  out += (me % SIZEY);

  for (i32 i = 0; i < MIDDLE; ++i) { out[i * (OUT_WG * OUT_SPACING)] = u[i]; }
}

// Carry propagation with optional MUL-3, over CARRY_LEN words.
// Input is conjugated and inverse-weighted.

//{{ CARRYA
KERNEL(G_W) NAME(P(Word2) out, CP(T2) in, P(CarryABM) carryOut, CP(T2) A, CP(u32) extras, P(u32) roundOut, P(u32) carryStats) {
  ENABLE_MUL2();
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;

  CarryABM carry = 0;
  
  u32 roundMax = 0;
  u32 carryMax = 0;
  
  u32 extra = reduce(extras[G_W * CARRY_LEN * gy + me] + (u32) (2u * BIG_HEIGHT * G_W * (u64) STEP % NWORDS) * gx % NWORDS);
  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;
    bool b1 = isBigWord(extra);
    bool b2 = isBigWord(reduce(extra + STEP));
    T2 x = conjugate(in[p]) * A[p];
    
#if ROUNDOFF
    roundMax = max(roundMax, roundoff(x));
#endif
    
#if DO_MUL3
    out[p] = unweightAndCarryMul(x, &carry, b1, b2, carry, &carryMax);
#else
    out[p] = unweightAndCarry(x, &carry, b1, b2, carry, &carryMax);
#endif
    extra = reduce(extra + (u32) (2u * STEP % NWORDS));
  }
  carryOut[G_W * g + me] = carry;

#if ROUNDOFF
  roundMax = work_group_reduce_max(roundMax);
  carryMax = work_group_reduce_max(carryMax);
  if (me == 0) {
    atomic_max(&roundOut[4], roundMax);
    atomic_max(&carryStats[4], carryMax);
    u32 oldCount = atomic_inc(&roundOut[5]);
    assert(oldCount < get_num_groups(0));
    if (oldCount == get_num_groups(0) - 1) {
      roundMax = atomic_xchg(&roundOut[4], 0);
      carryMax = atomic_xchg(&carryStats[4], 0);
      roundOut[5] = 0;
      
      atom_add((global ulong *) &roundOut[0], roundMax);
      atomic_max(&roundOut[2], roundMax);
      atomic_inc(&roundOut[3]);
      
      atom_add((global ulong *) &carryStats[0], carryMax);
      atomic_max(&carryStats[2], carryMax);
      atomic_inc(&carryStats[3]);
    }
  }
#endif
}
//}}

//== CARRYA NAME=carryA,DO_MUL3=0
//== CARRYA NAME=carryM,DO_MUL3=1

KERNEL(G_W) carryB(P(Word2) io, CP(CarryABM) carryIn, CP(u32) extras) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);  
  u32 gx = g % NW;
  u32 gy = g / NW;

  ENABLE_MUL2();

  u32 extra = reduce(extras[G_W * CARRY_LEN * gy + me] + (u32) (2u * BIG_HEIGHT * G_W * (u64) STEP % NWORDS) * gx % NWORDS);
  
  u32 step = G_W * gx + WIDTH * CARRY_LEN * gy;
  io += step;

  u32 HB = BIG_HEIGHT / CARRY_LEN;

  u32 prev = (gy + HB * G_W * gx + HB * me + (HB * WIDTH - 1)) % (HB * WIDTH);
  u32 prevLine = prev % HB;
  u32 prevCol  = prev / HB;

  CarryABM carry = carryIn[WIDTH * prevLine + prevCol];

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = i * WIDTH + me;
    io[p] = carryWord(io[p], &carry, isBigWord(extra), isBigWord(reduce(extra + STEP)));
    if (!carry) { return; }
    extra = reduce(extra + (u32) (2u * STEP % NWORDS));
  }
}

void release() {
#if 0
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE, memory_order_release, memory_scope_device);
  work_group_barrier(0);
#else
  work_group_barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE, memory_scope_device);
#endif
}

void acquire() {
#if 0
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
  work_group_barrier(0);
#else
  work_group_barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE, memory_scope_device);
#endif
}

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway" carry data forwarding from one group to the next.
// See tools/expand.py for the meaning of '//{{', '//}}', '//==' -- a form of macro expansion
//{{ CARRY_FUSED
KERNEL(G_W) NAME(P(T2) out, CP(T2) in, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
                 CP(u32) bits, CP(T2) groupWeights, CP(T2) threadWeights, P(u32) roundOut, P(u32) carryStats) {
  local T2 lds[WIDTH / 2];
  
  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
  u32 line = gr % H;

  T2 u[NW];
  
  readCarryFusedLine(in, u, line);

#if NW == 4
  u32 b = bits[G_W / 4 * line + me / 4];
  b = b >> ((me & 3) * 8);
#else
  u32 b = bits[G_W / 2 * line + me / 2];
  b = b >> ((me & 1) * 16);
#endif

  ENABLE_MUL2();
  fft_WIDTH(lds, u, smallTrig);

// Convert each u value into 2 words and a 32 or 64 bit carry

  Word2 wu[NW];
  T2 weights = groupWeights[line] * threadWeights[me];
  T invWeight = weights.x;
#if CF_MUL
  P(CFMcarry) carryShuttlePtr = (P(CFMcarry)) carryShuttle;
  CFMcarry carry[NW+1];
#else
  P(CFcarry) carryShuttlePtr = (P(CFcarry)) carryShuttle;
  CFcarry carry[NW+1];
#endif

  u32 roundMax = 0;
  u32 carryMax = 0;
  
  for (i32 i = 0; i < NW; ++i) {
    invWeight = optionalDouble(invWeight);
    T invWeight2 = optionalDouble(invWeight * IWEIGHT_STEP);
    T2 x = conjugate(u[i]) * U2(invWeight, invWeight2);

#if ROUNDOFF
    roundMax = max(roundMax, roundoff(x));
#endif
    
#if CF_MUL    
    wu[i] = unweightAndCarryMul(x, &carry[i], test(b, 2 * i), test(b, 2 * i + 1), 0, &carryMax);
#else
    wu[i] = unweightAndCarry(x, &carry[i], test(b, 2 * i), test(b, 2 * i + 1), 0, &carryMax);
#endif
    invWeight *= IWEIGHT_BIGSTEP;
  }

  if (gr < H) {
    for (i32 i = 0; i < NW; ++i) {
#if OLD_CARRY_LAYOUT
      carryShuttlePtr[gr * WIDTH + i * G_W + me] = carry[i];
#else
      // Write the carries to carry shuttle.  AMD GPUs are faster writing and reading 4 consecutive values at a time.
      // However, seemingly innocuous code changes can affect VGPR usage which if it changes occupancy can be a
      // more important consideration.
      carryShuttlePtr[gr * WIDTH + me * NW + i] = carry[i];
#endif
    }
  }
    
  release();

  // Signal that this group is done writing the carry.
  if (gr < H && me == 0) {
#ifdef ATOMICALLY_CORRECT
    atomic_store((atomic_uint *) &ready[gr], 1);
#else
    ready[gr] = 1;
#endif
  }

  if (gr == 0) { return; }

#if ROUNDOFF
  roundMax = work_group_reduce_max(roundMax);
  carryMax = work_group_reduce_max(carryMax);
  if (me == 0) {
    // Roundout 0/1 = sum(iteration_maxerr)
    // Roundout 2   = max(iteration_maxerr)
    // Roundout 3   = count(iteration)
    // Roundout 4   = max(workgroup maxerr)
    // Roundout 5   = count(workgroup)
    atomic_max(&roundOut[4], roundMax);
    atomic_max(&carryStats[4], carryMax);
    u32 oldCount = atomic_inc(&roundOut[5]);
    assert(oldCount < BIG_HEIGHT);
    if (oldCount == BIG_HEIGHT - 1) {
      roundMax = atomic_xchg(&roundOut[4], 0);
      carryMax = atomic_xchg(&carryStats[4], 0);
      roundOut[5] = 0;
      
      atom_add((global ulong *) &roundOut[0], roundMax);
      atomic_max(&roundOut[2], roundMax);
      atomic_inc(&roundOut[3]);
      
      atom_add((global ulong *) &carryStats[0], carryMax);
      atomic_max(&carryStats[2], carryMax);
      atomic_inc(&carryStats[3]);
    }
  }
#endif // ROUNDOFF

  // Wait until the previous group is ready with the carry.
  if (me == 0) {
    while(!atomic_load((atomic_uint *) &ready[gr - 1]));
    ready[gr - 1] = 0; // atomic_store((atomic_uint *) &ready[gr - 1], 0);
  }

  acquire();

  // Read from the carryShuttle carries produced by the previous WIDTH row.  Rotate carries from the last WIDTH row.
  // The new carry layout lets the compiler generate global_load_dwordx4 instructions.

  assert(gr > 0 && gr <= H);
  
#if OLD_CARRY_LAYOUT
  if (gr == H) {
    for (i32 i = 0; i < NW; ++i) {
      carry[i] = carryShuttlePtr[(gr - 1) * WIDTH + ((i * G_W + (WIDTH - 1) + me) % WIDTH)];
    }
  } else {
    for (i32 i = 0; i < NW; ++i) {
      carry[i] = carryShuttlePtr[(gr - 1) * WIDTH + i * G_W + me];
    }
  }
#else
  if (gr < H) {
    for (i32 i = 0; i < NW; ++i) {
      carry[i] = carryShuttlePtr[(gr - 1) * WIDTH + me * NW + i];
    }
  } else if (gr == H) {
    for (i32 i = 0; i < NW; ++i) {
      carry[i] = carryShuttlePtr[(gr - 1) * WIDTH + (me + G_W - 1) % G_W * NW + i];
    }
    if (me == 0) {
      carry[NW] = carry[NW-1];
      for (i32 i = NW-1; i; --i) { carry[i] = carry[i-1]; }
      carry[0] = carry[NW];
    }
  } else {
    // This is unreachable because 'gr' is in range [1 .. H], so one of the previous branches must have been taken
    // assert(gr <= H);
  }
#endif

  // Apply each 32 or 64 bit carry to the 2 words and weight the result to create new u values.

  T weight = weights.y;
  for (i32 i = 0; i < NW; ++i) {
    weight = optionalHalve(weight);
    T weight2 = optionalHalve(weight * WEIGHT_STEP);

    u[i] = carryAndWeightFinal(wu[i], carry[i], U2(weight, weight2), test(b, 2 * i));
    weight *= WEIGHT_BIGSTEP;
  }

  bar();
  fft_WIDTH(lds, u, smallTrig);

  write(G_W, NW, u, out, WIDTH * line);
}
//}}

//== CARRY_FUSED NAME=carryFused,    CF_MUL=0
//== CARRY_FUSED NAME=carryFusedMul, CF_MUL=1

// from transposed to sequential.
KERNEL(256) transposeOut(P(Word2) out, CP(Word2) in) {
  local Word2 lds[4096];
  ENABLE_MUL2();
  transposeWords(WIDTH, BIG_HEIGHT, lds, in, out);
}

// from sequential to transposed.
KERNEL(256) transposeIn(P(Word2) out, CP(Word2) in) {
  local Word2 lds[4096];
  ENABLE_MUL2();
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

KERNEL(SMALL_HEIGHT / 2 / 4) square(P(T2) out, CP(T2) in) {
  u32 W = SMALL_HEIGHT;
  u32 H = ND / W;

  ENABLE_MUL2();

  u32 me = get_local_id(0);
  u32 line1 = get_group_id(0);
  u32 line2 = (H - line1) % H;
  u32 g1 = transPos(line1, MIDDLE, WIDTH);
  u32 g2 = transPos(line2, MIDDLE, WIDTH);

  T2 base_squared = slowTrig(me * H + line1, W * H / 2, W * H / 4);
  T2 step = U2(M_SQRT1_2, -M_SQRT1_2); // trig(pi/4)
  
  for (u32 i = 0; i < 4; ++i, base_squared = mul(base_squared, step)) {
    if (i == 0 && line1 == 0 && me == 0) {
      out[0]     = foo_m2(conjugate(in[0]));
      out[W / 2] = 4 * sq(conjugate(in[W / 2]));
    } else {
      u32 k = g1 * W + i * (W / 8) + me;
      u32 v = g2 * W + (W - 1) + (line1 == 0) - i * (W / 8) - me;
      T2 a = in[k];
      T2 b = in[v];
      onePairSq(a, b, swap_squared(base_squared));
      out[k] = a;
      out[v] = b;
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
  onePairMul(a, b, c, d, swap_squared(slowTrig(me * H + line1, W * H / 2, W * H / 4)));
  io[k] = a;
  io[v] = b;
}
//}}

//== MULTIPLY NAME=multiply, MULTIPLY_DELTA=0

#if NO_P2_FUSED_TAIL
//== MULTIPLY NAME=multiplyDelta, MULTIPLY_DELTA=1
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
  readTailFusedLine(in, u, line1, memline1);
  readTailFusedLine(in, v, line2, memline2);
  fft_HEIGHT(lds, u, smallTrig1);
  bar();
  fft_HEIGHT(lds, v, smallTrig1);
#endif

  u32 me = get_local_id(0);
  if (line1 == 0) {
    // Line 0 is special: it pairs with itself, offseted by 1.
    reverse(G_H, lds, u + NH/2, true);    
    pairSq(NH/2, u,   u + NH/2, slowTrig(me, W / 2, W / 4), true);
    reverse(G_H, lds, u + NH/2, true);

    // Line H/2 also pairs with itself (but without offset).
    reverse(G_H, lds, v + NH/2, false);
    pairSq(NH/2, v,   v + NH/2, slowTrig(1 + 2 * me, W, W / 2), false);
    reverse(G_H, lds, v + NH/2, false);
  } else {    
    reverseLine(G_H, lds, v);
    pairSq(NH, u, v, slowTrig(line1 + me * H, ND / 2, ND / 4), false);
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
  readTailFusedLine(in, u, line1, memline1);
  readTailFusedLine(in, v, line2, memline2);
  readDelta(G_H, NH, p, a, b, memline1 * SMALL_HEIGHT);
  readDelta(G_H, NH, q, a, b, memline2 * SMALL_HEIGHT);
  fft_HEIGHT(lds, u, smallTrig1);
  bar();
  fft_HEIGHT(lds, v, smallTrig1);
#elif MUL_LOW
  readTailFusedLine(in, u, line1, memline1);
  readTailFusedLine(in, v, line2, memline2);
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
  readTailFusedLine(in, u, line1, memline1);
  readTailFusedLine(in, v, line2, memline2);
  readTailFusedLine(a, p, line1, memline1);
  readTailFusedLine(a, q, line2, memline2);
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
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, slowTrig(me, W / 2, W / 4), true);
    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);

    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, slowTrig(1 + 2 * me, W, W / 2), false);
    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
  } else {    
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);
    pairMul(NH, u, v, p, q, slowTrig(line1 + me * H, W * H / 2, W * H / 4), false);
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

// Generate a small unused kernel so developers can look at how well individual macros assemble and optimize
#ifdef TEST_KERNEL
KERNEL(256) testKernel(global double* io) {
  u32 me = get_local_id(0);
  io[me] = kcos(io[me]);
}
#endif

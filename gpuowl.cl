// Copyright Mihai Preda and George Woltman.

/* List of user-serviceable -use flags and their effects

OUT_WG,OUT_SIZEX,OUT_SPACING <AMD default is 256,32,4> <nVidia default is 256,4,1 but needs testing>
IN_WG,IN_SIZEX <AMD default is 256,32>  <nVidia default is 256,4 but needs testing>

UNROLL_WIDTH <nVidia default>
NO_UNROLL_WIDTH <AMD default>

OLD_FFT8 <default>
NEWEST_FFT8
NEW_FFT8

OLD_FFT5
NEW_FFT5 <default>
NEWEST_FFT5

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
#define TRIG_COMPUTE 1
#endif

#define STR(x) XSTR(x)
#define XSTR(x) #x

#define OVL __attribute__((overloadable))
#define VECTOR(n) __attribute__((ext_vector_type(n)))

#pragma OPENCL FP_CONTRACT ON

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// 64-bit atomics used in kernel sum64
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

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
#error WIDTH
#endif

#if SMALL_HEIGHT == 1024 || SMALL_HEIGHT == 256
#define NH 4
#else
#error SMALL_HEIGHT
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

void bar() { barrier(0); }

typedef int i32;
typedef uint u32;
typedef long i64;
typedef ulong u64;
typedef long long i128;
typedef unsigned long long u128;

typedef i64 Word;
typedef VECTOR(2) Word Word2;
typedef i64 Carry;

typedef i128 T;
typedef VECTOR(2) T T2;

typedef u128 Weight;
typedef VECTOR(2) Weight Weight2;

u32  U32(u32 x)   { return x; }
u64  U64(u64 x)   { return x; }
u128 U128(u128 x) { return x; }
i32  I32(i32 x)   { return x; }
u64  I64(i64 x)   { return x; }
u128 I128(i128 x) { return x; }

u32 hiU32(u64 x) { return x >> 32; }
u64 hiU64(u128 x) { return x >> 64; }

#define SHR(a, shift) a = (a >> shift)

// u64  mul64 (u64 a, u64 b)   { return a * b; }
// u128 mul128(u128 a, u128 b) { return a * b; }

u32  mulh32 (u32 a, u32 b)   { return hiU32(U64(a) * b); }
u64  mulh64 (u64 a, u64 b)   { return (a >> 32) * (b >> 32) + mulh32(a >> 32, b) + mulh32(a, b >> 32); }
u128 mulh128(u128 a, u128 b) { return (a >> 64) * (b >> 64) + mulh64(a >> 64, b) + mulh64(a, b >> 64); }

u128 OVL mulShl(u128 a, u128 b, u32 shift) {
  u32 n1 = clz(U32(a >> 96));
  u32 n2 = clz(U32(b >> 96));
  assert(n1 + n2 >= shift);
  return mulh128(a << n1, b << n2) >> (n1 + n2 - shift);
}

i128 OVL mulShl(i128 a, i128 b, u32 shift) {
  bool neg1 = a < 0;
  bool neg2 = b < 0;
  if (neg1) { a = -a; }
  if (neg2) { b = -b; }
  u128 r = mulShl((u128) a, (u128) b, shift);
  return (neg1 != neg2) ? -r : r;
}

u128 OVL mul(u128 a, u128 b) { return mulShl(a, b, 0); }
i128 OVL mul(i128 a, i128 b) { return mulShl(a, b, 0); }

i128 OVL mul(i128 a, u128 b) {
  bool neg = a < 0;
  if (neg) { a = -a; } // a = abs(a);
  i128 r = mul((u128) a, b);
  return neg ? -r : r;
}

// u128 OVL mulSimple(u128 a, u128 b) { return mulh128(a, b); }

u128 OVL sqShl(u128 a, u32 shift) { return mulShl(a, a, shift); }
u128 OVL sqShl(i128 a, u32 shift) { return mulShl(a, a, shift); }

u128 OVL sq(u128 a) { return sqShl(a, 0); }
u128 OVL sq(i128 a) { return sqShl(a, 0); }

// ---- Complex ----

T2 OVL sq(T2 a) { return (T2) ((T) (sq(a.x) - sq(a.y)), mulShl(a.x, a.y, 1)); }
T2 OVL sqShl(T2 a, u32 shift) { return (T2)((T)(sqShl(a.x, shift) - sqShl(a.y, shift)), mulShl(a.x, a.y, shift + 1)); }

T2 OVL mulShl(T2 a, T2 b, u32 shift) { return (T2) (mulShl(a.x, b.x, shift) - mulShl(a.y, b.y, shift), mulShl(a.x, b.y, shift) + mulShl(a.y, b.x, shift)); }
T2 OVL mul(T2 a, T2 b) { return mulShl(a, b, 0); }

T2 OVL mul(T2 a, T factor) { return (T2) (mul(a.x, factor), mul(a.y, factor)); }

T2 mad(T2 a, T2 b, T2 c) { return mul(a, b) + c; }


// ---- Bits ----

bool test(u32 bits, u32 pos) { return (bits >> pos) & 1; }

#define STEP (NWORDS - (EXP % NWORDS))
// bool isBigWord(u32 extra) { return extra < NWORDS - STEP; }

#define SMALL_BITS (EXP / NWORDS)
#define BIG_BITS (SMALL_BITS + 1)

u32 bitlen(bool b) { return SMALL_BITS + b; }


// ---- Trig ----

T2 mul_t4(T2 a)  { return (T2) (a.y, -a.x); } // mul(a, U2( 0, -1)); }

#define SQRT1_2 0xb504f333f9de6484597d89b3754abe9fULL

T2 mul_t8 (T2 a) { return mul((T2) (a.y + a.x, a.y - a.x), SQRT1_2); }
T2 mul_3t8(T2 a) { return mul((T2) (a.y - a.x, a.y + a.x), SQRT1_2); }

T2 swap(T2 a)      { return (T2) (a.y,  a.x); }
T2 conjugate(T2 a) { return (T2) (a.x, -a.y); }


// ---- Weight ----

#define BITBUFFER 1
#define SHIFTDOWN (30 + BITBUFFER)

// Flush a word of at most BIG_BITS up, keeping BITBUFFER extra bits of buffer
T OVL shiftUp(Word u) {
  assert(BIG_BITS <= (64 - BITBUFFER));
  return U128(u << (64 - BITBUFFER - BIG_BITS)) << 64;
}

T2 OVL shiftUp(Word2 u) { return (T2) (shiftUp(u.x), shiftUp(u.y)); }

// Round and shift down. The return may not fit on a Word due to unpropagated carry, thus return is i128.
i128 shiftDown(T u) {
  u >>= (SHIFTDOWN - 1);
  return (u >> 1) + (u & 1);
}

// T OVL weight(Word a, Weight w) { return mul(asT(a), w); }
// T2 OVL weight(Word2 a, Weight2 w) { return (T2) (weight(a.x, w.x), weight(a.y, w.y)); }

// This routine works for both forward und inverse weight updating.
// Forward weighs are represented halved, and must always have the leading bit 1 (w in [0.5, 1))
// Inverse weights must also have the leading bit 1 (iw in (0.5, 1])
Weight updateWeight(Weight w, Weight step) {
  assert((w >> 127) && (step >> 127));
  w = mulh128(w, step);
  if (!(w >> 127)) { w <<= 1; }
  assert(w >> 127);
  return w;
}

Weight2 updateWeights(Weight2 w, Weight2 step) { return (Weight2) (updateWeight(w.x, step.x), updateWeight(w.y, step.y)); }

T2 mulWeight(T2 u, Weight2 w) { return (T2) (mul(u.x, w.x), mul(u.y, w.y)); }

i64 lowBits(i64 u, u32 n) { return (u << (64 - n)) >> (64 - n); }

#define SHIFTDOWN (30 + BITBUFFER)

Word carryStep(T u, Carry* outCarry, bool isBig) {
  u32 nBits = bitlen(isBig);
  assert(nBits < 64);
  assert((u >> (64 + nBits)) == 0 || (u >> (64 + nBits)) == -1); // check outCarry fits on 64bits
  
  Word w = lowBits(u, nBits);
  u >>= nBits;
  assert(I64(u) == u);
  *outCarry = I64(u) + (w < 0);
  return w;
}

Word2 carryPair(T2 u, Carry* outCarry, bool b1, bool b2, i64 inCarry) {
  Carry midCarry;
  Word a = carryStep(shiftDown(u.x) + inCarry, &midCarry, b1);
  Word b = carryStep(shiftDown(u.y) + midCarry, outCarry, b2);
  return (Word2) (a, b);
}

Word2 carryFinal(Word2 u, Carry inCarry, bool b1) {
  Carry tmpCarry;
  u.x = carryStep(u.x + inCarry, &tmpCarry, b1);
  u.y += tmpCarry;
  return u;
}

// Carry propagation from word and carry.
Word2 carryWord(Word2 a, Carry* carry, bool b1, bool b2) {
  a.x = carryStep(a.x + *carry, carry, b1);
  a.y = carryStep(a.y + *carry, carry, b2);
  return a;
}

// Propagate carry this many pairs of words.
#define CARRY_LEN 8

T2 addsub(T2 a) { return (T2) (a.x + a.y, a.x - a.y); }

T2 fooShl(T2 a, u32 shift) {
  T c = sqShl(a.x + a.y, shift);
  T d = mulShl(a.x, a.y, shift + 1);
  return (T2)(c - d, d);
}

// fooShl(a, s) == foo2Shl(a, a, s-1).
T2 foo2Shl(T2 a, T2 b, u32 shift) {
  a = addsub(a);
  b = addsub(b);
  return addsub((T2) (mulShl(a.x, b.x, shift), mulShl(a.y, b.y, shift)));
}

// computes 2*[x^2+y^2 + i*(2*x*y)]. Needs a name. foo(a) == foo2(a, a).
T2 foo(T2 a) { return fooShl(a, 1); }
T2 foo_m2(T2 a) { return fooShl(a, 2); }

// computes 2*[a.x*b.x+a.y*b.y + i*(a.x*b.y+a.y*b.x)]
T2 foo2(T2 a, T2 b) { return foo2Shl(a, b, 0); }
T2 foo2_m2(T2 a, T2 b) { return foo2Shl(a, b, 1); }

#define X2(a, b) { T2 t = a; a += b; b = t - b; } // { T2 t = a; a.x += b.x; b.x = t.x - b.x; a.y += b.y; b.y = t.y - b.y; }

// Same as X2(a, b), b = mul_t4(b)
#define X2_mul_t4(a, b) { T2 t = a; a += b; t.x = b.x - t.x; b.x = t.y - b.y; b.y = t.x; } // { T2 t = a; a.x += b.x; a.y += b.y; t.x = b.x - t.x; b.x = t.y - b.y; b.y = t.x; }

// Same as X2(a, conjugate(b))
#define X2conjb(a, b) { T2 t = a; a.x += b.x; a.y -= b.y; b.x = t.x - b.x; b.y += t.y; }

// Same as X2(a, b), a = conjugate(a)
#define X2conja(a, b) { T2 t = a; a.x += b.x; a.y = -a.y - b.y; b = t - b; }

#define SWAP(a, b) { T2 t = a; a = b; b = t; }

void fft4(T2 *u) {
  for (u32 i = 0; i < 4; ++i) { SHR(u[i], 2); }
  
  X2(u[0], u[2]);
  X2_mul_t4(u[1], u[3]);
  T2 t = u[2];
  u[2] = u[0] - u[1];
  u[0] = u[0] + u[1];
  u[1] = t + u[3];
  u[3] = t - u[3];  
}

void fft2(T2* u) {
  SHR(u[0], 1);
  SHR(u[1], 1);

  X2(u[0], u[1]);
}

#if 0
void fft3by(T2* u, u32 incr) {
  SHR(u[0], 2);
  SHR(u[incr], 2);
  SHR(u[2 * incr], 2);

  // const double COS1 = -0.5;					// cos(tau/3), -0.5
  const double SIN1 = 0.86602540378443864676372317075294;	// sin(tau/3), sqrt(3)/2, 0.86602540378443864676372317075294
  
  X2_mul_t4(u[incr], u[2 * incr]);				// (r2+r3 i2+i3),  (i2-i3 -(r2-r3))
  
  T2 tmp23 = u[0] - (u[incr] >> 1); // COS1 * u[1*incr];
  
  u[0] += u[incr];

  u[incr] = tmp23;
  
  u[2 * incr] = mul(u[2 * incr], SIN1);
  X2(u[incr], u[2 * incr]);
}

void fft3(T2 *u) { fft3by(u, 1); }
#endif

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

void tabMul(u32 WG, const global T2 *trig, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  for (i32 i = 1; i < n; ++i) {
    u[i] = mul(u[i], trig[me / f + i * (WG / f)]);
  }
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
    if (s != 4) { bar(); }
    fft4(u);
    shuflAndMul(64, lds, trig, u, 4, 1 << s);
  }
  fft4(u);
}

void fft256h(local T2 *lds, T2 *u, const global T2 *trig) {
  u32 me = get_local_id(0);
  fft4(u);
  for (int i = 0; i < 3; ++i) { u[1 + i] = mul(u[1 + i], trig[64 + 64 * i + me]); }
  shufl2(64, lds,  u, 4, 1);
  bar();
  fft4(u);
  shuflAndMul2(64, lds, trig, u, 4, 4);
  bar();
  fft4(u);
  shuflAndMul2(64, lds, trig, u, 4, 16);
  fft4(u);
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

// ---- Trig ----

// N represents a full circle, so N/2 is Pi radians and N/8 is Pi/4 radians.

global T2 TRIG_2SH[SMALL_HEIGHT / 4 + 1];
global T2 TRIG_BH[BIG_HEIGHT / 8 + 1];

#if TRIG_COMPUTE == 0
global T2 TRIG_N[ND / 8 + 1];
#elif TRIG_COMPUTE == 1
global T2 TRIG_W[WIDTH / 2 + 1];
#endif

Weight2 THREAD_WEIGHTS[G_W];
Weight2 CARRY_WEIGHTS[BIG_HEIGHT / CARRY_LEN];

T2 tableTrig(u32 k, u32 n, u32 kBound, global T2* trigTable) {
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

  T2 r = trigTable[k];

  if (flip) { r = -swap(r); }
  if (negateCos) { r.x = -r.x; }
  if (negate) { r = -r; }
  return r;
}

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

KERNEL(64) writeGlobals(global T2* trig2Sh, global T2* trigBh, global T2* trigN, global T2* trigW,
                        global Weight2* threadWeights, global Weight2* carryWeights
                        ) {
  for (u32 k = get_global_id(0); k < 2 * SMALL_HEIGHT/8 + 1; k += get_global_size(0)) { TRIG_2SH[k] = trig2Sh[k]; }
  for (u32 k = get_global_id(0); k < BIG_HEIGHT/8 + 1; k += get_global_size(0)) { TRIG_BH[k] = trigBh[k]; }

#if TRIG_COMPUTE == 0
  for (u32 k = get_global_id(0); k < ND/8 + 1; k += get_global_size(0)) { TRIG_N[k] = trigN[k]; }
#elif TRIG_COMPUTE == 1
  for (u32 k = get_global_id(0); k <= WIDTH/2; k += get_global_size(0)) { TRIG_W[k] = trigW[k]; }
#endif

  // Weights
  for (u32 k = get_global_id(0); k < G_W; k += get_global_size(0)) { THREAD_WEIGHTS[k] = threadWeights[k]; }
  for (u32 k = get_global_id(0); k < BIG_HEIGHT / CARRY_LEN; k += get_global_size(0)) { CARRY_WEIGHTS[k] = carryWeights[k]; }  
}

T2 slowTrig_2SH(u32 k, u32 kBound) { return tableTrig(k, 2 * SMALL_HEIGHT, kBound, TRIG_2SH); }
T2 slowTrig_BH(u32 k, u32 kBound)  { return tableTrig(k, BIG_HEIGHT, kBound, TRIG_BH); }

// Returns e^(-i * tau * k / n), (tau == 2*pi represents a full circle). So k/n is the ratio of a full circle.
// Inverse trigonometric direction is chosen as an FFT convention.
T2 slowTrig_N(u32 k, u32 kBound)   {
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

#if TRIG_COMPUTE == 1
  u32 a = (k + WIDTH/2) / WIDTH;
  i32 b = k - a * WIDTH;  
  T2 cs2 = TRIG_W[abs(b)];
  cs2.y = (b < 0) ? -cs2.y : cs2.y;
  T2 r = mul(TRIG_BH[a], cs2);
#elif TRIG_COMPUTE == 0
  double2 r = TRIG_N[k];
#else
#error set TRIG_COMPUTE to 0 or 1.
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
typedef CP(T2) Trig;

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
void readTailFusedLine(CP(T2) in, T2 *u, u32 line) {
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
  fft_HEIGHT(lds, u, smallTrig);
  write(G_H, NH, u, io, 0);
}

T fweightStep(u32 i) {
  const T TWO_TO_NTH[8] = {
    // 2^(k/8) -1 for k in [0..8)
    0
    /*
    0.090507732665257662,
    0.18920711500272105,
    0.29683955465100964,
    0.41421356237309503,
    0.54221082540794086,
    0.68179283050742912,
    0.83400808640934243,
    */
  };
  return TWO_TO_NTH[i * STEP % NW * (8 / NW)];
}

T iweightStep(u32 i) {
  const T TWO_TO_MINUS_NTH[8] = {
    // 2^-(k/8) - 1 for k in [0..8)
    0
    /*
    -0.082995956795328771,
    -0.15910358474628547,
    -0.2288945872960296,
    -0.29289321881345248,
    -0.35158022267449518,
    -0.40539644249863949,
    -0.45474613366737116,
    */
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

  T base = updateWeight(THREAD_WEIGHTS[me].y, CARRY_WEIGHTS[g / CARRY_LEN].y);
  base = updateWeight(base, fweightUnitStep(g % CARRY_LEN));

  for (u32 i = 0; i < NW; ++i) {
    T w1 = i == 0 ? base : updateWeight(base, fweightStep(i));
    T w2 = updateWeight(w1, WEIGHT_STEP);
    u32 p = G_W * i + me;
    u[i].x = mul(shiftUp(in[p].x), w1);
    u[i].y = mul(shiftUp(in[p].y), w2);
  }

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
#else
#error UNRECOGNIZED MIDDLE
#endif
}

void middleMul(T2 *u, u32 y, Trig trig) {
  assert(y < SMALL_HEIGHT);

  T2 w = slowTrig_BH(y, SMALL_HEIGHT);
  T2 step = w;
  
  for (u32 i = 1; i < MIDDLE; ++i) {
    u[i] = mul(u[i], w);
    w = mul(w, step);
  }
  
  // for (u32 i = 1; i < MIDDLE; ++i) { u[i] = mul(u[i], slowTrig_BH(y * i, SMALL_HEIGHT * i)); }
}

void middleMul2(T2 *u, u32 x, u32 y) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);

  T2 w = slowTrig_N(x * y, ND / MIDDLE);

  T2 step = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE);

  for (u32 i = 0; i < MIDDLE; ++i) {
    u[i] = mul(u[i], w);
    w = mul(w, step);
  }
}

void middleMul2Factor(T2 *u, u32 x, u32 y) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);

  T2 w = slowTrig_N(x * y, ND / MIDDLE);

  // TODO
  // u128 factor = -1;
  // u32 shift = 0;
  // w = mul(w, factor, shift);
  
  T2 step = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE);
  
  for (u32 i = 0; i < MIDDLE; ++i) {
    u[i] = mul(u[i], w);
    w = mul(w, step);
  }
}

#define MIDDLE_LDS_LIMIT 4

void middleShuffle(local long *lds, T2 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);

  if (MIDDLE <= MIDDLE_LDS_LIMIT / 2) {
    local long* p1 = lds + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    local long* p2 = lds + me;
    long4 *pu = (long4 *)u;

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].x; }
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize + MIDDLE * workgroupSize] = pu[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].x = p2[workgroupSize * i]; }
    for (int i = 0; i < MIDDLE; ++i) { pu[i].y = p2[workgroupSize * i + MIDDLE * workgroupSize]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].z; }
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize + MIDDLE * workgroupSize] = pu[i].w; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].z = p2[workgroupSize * i]; }
    for (int i = 0; i < MIDDLE; ++i) { pu[i].w = p2[workgroupSize * i + MIDDLE * workgroupSize]; }

  } else if (MIDDLE <= MIDDLE_LDS_LIMIT) {
    local long* p1 = lds + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    local long* p2 = lds + me;
    long4 *pu = (long4 *)u;

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].y = p2[workgroupSize * i]; }
    bar();

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].z; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].z = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].w; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].w = p2[workgroupSize * i]; }
    
  } else {
    local int* p1 = ((local int*) lds) + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    local int* p2 = ((local int*) lds) + me;
    int8 *pu = (int8 *)u;

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].x = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].y = p2[workgroupSize * i]; }
    bar();

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].z; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].z = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].w; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].w = p2[workgroupSize * i]; }
    bar();

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].s4; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].s4 = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].s5; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].s5 = p2[workgroupSize * i]; }
    bar();

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].s6; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].s6 = p2[workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].s7; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].s7 = p2[workgroupSize * i]; }
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

  middleMul2(u, startx + mx, starty + my);

  fft_MIDDLE(u);

  middleMul(u, starty + my, trig);
  local long lds[IN_WG / 2 * (MIDDLE <= MIDDLE_LDS_LIMIT ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);

  out += gx * (BIG_HEIGHT * IN_SIZEX) + gy * (MIDDLE * IN_WG) + me;
  for (u32 i = 0; i < MIDDLE; ++i) { out[i * IN_WG] = u[i]; }
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
  u32 starty = gy * SIZEY;      // Each input row increases FFT element by BIG_HEIGHT
  in += starty * BIG_HEIGHT + startx;

  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = in[i * SMALL_HEIGHT + my * BIG_HEIGHT + mx]; }

  middleMul(u, startx + mx, trig);

  fft_MIDDLE(u);

  // FFT results come out multiplied by the FFT length (NWORDS).  Also, for performance reasons
  // weights and invweights are doubled meaning we need to divide by another 2^2 and 2^2.
  double factor = 1.0 / (4 * 4 * NWORDS);

  middleMul2Factor(u, starty + my, startx + mx);
  local long lds[OUT_WG / 2 * (MIDDLE <= MIDDLE_LDS_LIMIT ? 2 * MIDDLE : MIDDLE)];

  middleShuffle(lds, u, OUT_WG, OUT_SIZEX);

  out += gx * (MIDDLE * WIDTH * OUT_SIZEX);
  out += (gy / OUT_SPACING) * (MIDDLE * (OUT_WG * OUT_SPACING));
  out += (gy % OUT_SPACING) * SIZEY;
  out += (me / SIZEY) * (OUT_SPACING * SIZEY);
  out += (me % SIZEY);

  for (i32 i = 0; i < MIDDLE; ++i) { out[i * (OUT_WG * OUT_SPACING)] = u[i]; }
}

// Carry propagation with optional MUL-3, over CARRY_LEN words.
// Input arrives conjugated and inverse-weighted.

KERNEL(G_W) carryA(P(Word2) out, CP(T2) in, P(Carry) carryOut, CP(u32) bits, P(u32) roundOut, P(u32) carryStats) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;

  Carry carry = 0;  
  float roundMax = 0;
  u32 carryMax = 0;

  // Split 32 bits into CARRY_LEN groups of 2 bits.
#define GPW (16 / CARRY_LEN)
  u32 b = bits[(G_W * g + me) / GPW] >> (me % GPW * (2 * CARRY_LEN));
#undef GPW

  Weight base = updateWeight(THREAD_WEIGHTS[me].x, CARRY_WEIGHTS[gy].x);
  base = updateWeight(base, iweightStep(gx));  
  
  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;
    Weight w1 = i == 0 ? base : updateWeight(base, iweightUnitStep(i));
    Weight w2 = updateWeight(w1, IWEIGHT_STEP);
    // T2 x = in[p];
    T2 x = (T2) (mul(in[p].x,  w1), mul(-in[p].y, w2));
    
#if STATS
    roundMax = max(roundMax, roundoff(conjugate(in[p]), U2(w1, w2)));
#endif
    
    out[p] = carryPair(x, &carry, test(b, 2 * i), test(b, 2 * i + 1), carry);
  }
  carryOut[G_W * g + me] = carry;

#if STATS
  updateStats(roundMax, carryMax, roundOut, carryStats);
#endif
}

KERNEL(G_W) carryB(P(Word2) io, CP(Carry) carryIn, CP(u32) bits) {
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

  Carry carry = carryIn[WIDTH * prevLine + prevCol];

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = i * WIDTH + me;
    io[p] = carryWord(io[p], &carry, test(b, 2 * i), test(b, 2 * i + 1));
    if (!carry) { return; }
  }
}

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway" carry data forwarding from one group to the next.
KERNEL(G_W) carryFused(P(T2) out, CP(T2) in, P(Carry) carryShuttle, P(u32) ready, Trig smallTrig,
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
  
  fft_WIDTH(lds, u, smallTrig);

// Convert each u value into 2 words and a 32 or 64 bit carry

  Word2 wu[NW];
  Weight2 weights = updateWeights(CARRY_WEIGHTS[line / CARRY_LEN], THREAD_WEIGHTS[me]);
  weights = updateWeights(weights, (Weight2) (iweightUnitStep(line % CARRY_LEN), fweightUnitStep(line % CARRY_LEN)));

  Carry carry[NW+1];
  float roundMax = 0;
  u32 carryMax = 0;
  
  // Apply the inverse weights

  T invBase = weights.x;
  
  for (u32 i = 0; i < NW; ++i) {
    T invWeight1 = i == 0 ? invBase : updateWeight(invBase, iweightStep(i));
    T invWeight2 = updateWeight(invWeight1, IWEIGHT_STEP);

#if STATS
    roundMax = max(roundMax, roundoff(conjugate(u[i]), (Weight2) (invWeight1, invWeight2)));
#endif

    u[i] = mulWeight(conjugate(u[i]), (Weight2) (invWeight1, invWeight2));
  }

  // Generate our output carries
  for (i32 i = 0; i < NW; ++i) {
    wu[i] = carryPair(u[i], &carry[i], test(b, 2 * i), test(b, 2 * i + 1), 0);
  }

  // Write out our carries
  if (gr < H) {
    for (i32 i = 0; i < NW; ++i) {
      carryShuttle[gr * WIDTH + me * NW + i] = carry[i];
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
      carry[i] = carryShuttle[(gr - 1) * WIDTH + me * NW + i];
    }
  } else {
    for (i32 i = 0; i < NW; ++i) {
      carry[i] = carryShuttle[(gr - 1) * WIDTH + (me + G_W - 1) % G_W * NW + i];
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
  
  T base = weights.y;
  
  for (u32 i = 0; i < NW; ++i) {
    T weight1 = i == 0 ? base : updateWeight(base, fweightStep(i));
    T weight2 = updateWeight(weight1, WEIGHT_STEP);
    u[i] = mulWeight(shiftUp(wu[i]), (Weight2) (weight1, weight2));
  }

// Clear carry ready flag for next iteration

  bar();
  if (me == 0) ready[gr - 1] = 0;

// Now do the forward FFT and write results

  fft_WIDTH(lds, u, smallTrig);
  write(G_W, NW, u, out, WIDTH * line);
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
  b = mulShl(a, b, 1); \
  a = mul(b2, conjugate_t_squared) + sq(a); \
  X2conja(a, b); \
}

// From original code t = swap(base) and we need sq(conjugate(t)).  This macro computes sq(conjugate(t)) from base^2.
#define swap_squared(a) (-a)

void pairSq(u32 N, T2 *u, T2 *v, T2 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = foo_m2(conjugate(u[i]));
      v[i] = sqShl(conjugate(v[i]), 2);
    } else {
      onePairSq(u[i], v[i], -base_squared);
    }

    if (N == NH) {
      onePairSq(u[i+NH/2], v[i+NH/2], base_squared);
    }

    T2 new_base_squared = mul_t4(base_squared);
    onePairSq(u[i+NH/4], v[i+NH/4], -new_base_squared);

    if (N == NH) {
      onePairSq(u[i+3*NH/4], v[i+3*NH/4], new_base_squared);
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
  T2 tmp = mad(a, c, mul(mul(b, d), conjugate_t_squared)); \
  b = mad(b, c, mul(a, d)); \
  a = tmp; \
  X2conja(a, b); \
}

void pairMul(u32 N, T2 *u, T2 *v, T2 *p, T2 *q, T2 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = conjugate(foo2_m2(u[i], p[i]));
      v[i] = mulShl(conjugate(v[i]), conjugate(q[i]), 2);
    } else {
      onePairMul(u[i], v[i], p[i], q[i], -base_squared);
    }

    if (N == NH) {
      onePairMul(u[i+NH/2], v[i+NH/2], p[i+NH/2], q[i+NH/2], base_squared);
    }

    T2 new_base_squared = mul_t4(base_squared);
    onePairMul(u[i+NH/4], v[i+NH/4], p[i+NH/4], q[i+NH/4], -new_base_squared);

    if (N == NH) {
      onePairMul(u[i+3*NH/4], v[i+3*NH/4], p[i+3*NH/4], q[i+3*NH/4], new_base_squared);
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
    io[W / 2] = conjugate(mulShl(io[W / 2], inA[W / 2] - inB[W / 2], 2));
#else
    io[0]     = foo2_m2(conjugate(io[0]), conjugate(in[0]));
    io[W / 2] = conjugate(mulShl(io[W / 2], in[W / 2], 2));
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
 
// Generate a small unused kernel so developers can look at how well individual macros assemble and optimize
#ifdef TEST_KERNEL
 
KERNEL(256) testKernel(global long long* io) {
  u32 me = get_local_id(0);
  // ulong x = io[me].x;
  // ulong y = io[me].y;
  // io[me].x = (((unsigned long long) x) * y) >> 64u;
  /*
  long2 xy = as_long2(io[me]);
  long x = xy.x;
  long y = xy.y;
  
  // long long x = io[me];
  io[me] = ((long long) x) * y;
  */

  io[me] = io[me] + io[me + 1];
}
#endif


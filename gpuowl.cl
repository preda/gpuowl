// gpuOwl, an OpenCL Mersenne primality test.
// Copyright (C) 2017 Mihai Preda.

// The data is organized in pairs of words in a matrix WIDTH x HEIGHT.
// The pair (a, b) is sometimes interpreted as the complex value a + i*b.
// The order of words is column-major (i.e. transposed from the usual row-major matrix order).

// Expected defines: WIDTH, HEIGHT, EXP.
// One of: FFT_NTT (Number Theoretic Transform), FFT_FP (Floating Point).
// If FFT_NTT, also LOG_ROOT2 is expected, LOG_ROOT2 == log2(32 / (NWORDS % 31) % 31).
// If FFT_FP, one of: FP_SP (Single Precision) or FP_DP (Double Precision).

// FFT_NTT: a GF(P^2) (Galois Field) convolution, with P == M(31) == 2^31-1 a Mersenne prime.
// GF(P^2) means Gaussian integers ("complex integers") with the real/imaginary part modulo P.

// Number of words; a power of two.
#define NWORDS (WIDTH * HEIGHT * 2u)

// Used in bitlen() and weighting.
#define STEP (NWORDS - (EXP & (NWORDS - 1)))

// Each word has either BASE_BITLEN ("small word") or BASE_BITLEN+1 ("big word") bits.
#define BASE_BITLEN (EXP / NWORDS)

// Propagate carry this many pairs of words.
#define CARRY_LEN 16

// OpenCL 2.x introduces the "generic" memory space, so there's no need to specify "global" on pointers everywhere.
#if __OPENCL_C_VERSION__ >= 200
#define G
#else
#define G global
#endif


#ifdef FFT_NTT

typedef uint T;
typedef uint2 T2;

typedef uint Carry;
typedef uint2 Word2;

#else  // FP below.

#pragma OPENCL FP_CONTRACT ON

#ifdef FP_DP

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef double T;
typedef double2 T2;

#else

typedef float T;
typedef float2 T2;

#endif // FP_DP

typedef long Carry;
typedef int2 Word2;

#endif // FFT_NTT


// make a pair of Ts.
T2 U2(T a, T b) { return (T2)(a, b); }


#ifdef FFT_NTT

ulong u64(uint a) { return a; } // cast to 64 bits.

#include "nttshared.h"

#else // FP below.

T neg(T x) { return -x; }
T add1(T a, T b) { return a + b; }
T sub1(T a, T b) { return a - b; }

T2 add(T2 a, T2 b) { return a + b; }
T2 sub(T2 a, T2 b) { return a - b; }

T shl1(T a, uint k) { return a * (1 << k); }

// complex mul
T2 mul(T2 a, T2 b) { return U2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

// complex square
T2 sq(T2 a) { return U2((a.x + a.y) * (a.x - a.y), 2 * a.x * a.y); }

T mul1(T a, T b) { return a * b; }

#endif // FFT_NTT


T2 shl(T2 a, uint k) { return U2(shl1(a.x, k), shl1(a.y, k)); }

T2 addsub(T2 a) { return U2(add1(a.x, a.y), sub1(a.x, a.y)); }
T2 swap(T2 a) { return U2(a.y, a.x); }
T2 conjugate(T2 a) { return U2(a.x, neg(a.y)); }

uint extra(uint k) { return mul24(k, STEP) & (NWORDS - 1); }

void bar()    { barrier(CLK_LOCAL_MEM_FENCE); }
void bigBar() { barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); }

// Is the word at pos a big word (BASE_BITLEN+1 bits)? (vs. a small, BASE_BITLEN bits word).
bool isBigWord(uint k) { return extra(k) + STEP < NWORDS; }

// Number of bits for the word at pos.
uint bitlen(uint k) { return EXP / NWORDS + isBigWord(k); }


#ifdef FFT_NTT

// mul with (0, 1). (twiddle of tau/4, sqrt(-1) aka "i").
uint2 mul_t4(uint2 a) { return U2(neg(a.y), a.x); }

// mul with (2^15, 2^15). (twiddle of tau/8 aka sqrt(i)). Note: 2 * (2^15)^2 == 1 (mod M31).
uint2 mul_t8(uint2 a) { return U2(shl1(a.x + neg(a.y), 15), shl1(a.x + a.y, 15)); }

// mul with (-2^15, 2^15). (twiddle of 3*tau/8).
uint2 mul_3t8(uint2 a) { return U2(shl1(neg(a.x) + neg(a.y), 15), shl1(a.x + neg(a.y), 15)); }

#else // FP below.

T2 mul_t4(T2 a)  { return mul(a, U2( 0, -1)); }
T2 mul_t8(T2 a)  { return mul(a, U2( 1, -1)) * (T)(M_SQRT1_2); }
T2 mul_3t8(T2 a) { return mul(a, U2(-1, -1)) * (T)(M_SQRT1_2); }

#endif // FFT_NTT


#ifdef FFT_NTT

// NWORDS-th order root of 2: root2 ^ NWORDS == 2 (mod M31)
// LOG_ROOT2 == 32 / (NWORDS % 31) % 31
uint weight1(uint x, uint pos) { return shl1(x, (extra(pos) * LOG_ROOT2) % 31); }

// N * 2^(31 - LOG_NWORDS) == 1 (mod M31).
uint unweight1(uint x, uint pos) {
  x = (x + ((x + 1) >> 31)) & M31; // if x==M31, set it to 0.
  return shl1(x, (extra(pos) * (31 - LOG_ROOT2) + (31 - LOG_NWORDS - 2)) % 31 );
}

T2 weight(Word2 a, uint pos, const uint2 *dummyA, uint dummyP) { return U2(weight1(a.x, 2 * pos + 0), weight1(a.y, 2 * pos + 1)); }
Word2 unweight(T2 a, uint pos) { return (Word2) (unweight1(a.x, 2 * pos + 0), unweight1(a.y, 2 * pos + 1)); }

uint lowBits(uint x, uint bits) { return x & ((1 << bits) - 1); }

// one step of carry propagation.
uint carryStep(uint x, Carry *carry, uint bits) {
  x += *carry; //! must not modulo-reduce in carry propagation.
  *carry = x >> bits;
  return lowBits(x, bits);
}

uint carryStep3(uint x, Carry *carry, uint bits) {
  // Do the times3-plus-carry on extended bits to avoid 32-bit overflow.
  ulong a = u64(x) * 3 + *carry;
  *carry = a >> bits;
  return lowBits(a, bits);
}

uint update(uint x, Carry *carry, uint bits) { return carryStep(x, carry, bits); }

// Reverse weighting and carry propagation for a pair of words; with optional MUL-3.
Word2 car0(bool doMul3, T2 u, Carry *carry, uint pos, const T2 *dummyA, uint dummyP) {
  u = unweight(u, pos);
  if (doMul3) {
    u.x = carryStep3(u.x, carry, bitlen(2 * pos + 0));
    u.y = carryStep3(u.y, carry, bitlen(2 * pos + 1));
  } else {
    u.x = carryStep(u.x, carry, bitlen(2 * pos + 0));
    u.y = carryStep(u.y, carry, bitlen(2 * pos + 1));
  }
  return u;
}

#else

// Round(!) x to long.
long toLong(double x) { return rint(x); }

int lowBits(int u, uint bits) { return (u << (32 - bits)) >> (32 - bits); }

// carry step, keeping the reduced word and the carry in the same domain (e.g. double).
T carryStep(T x, T *carry, int bits) {
  x += *carry;
  *carry = rint(ldexp(x, -bits));
  return x - ldexp(*carry, bits);
}

// carry step, translating the reduced word and the carry to integrals (int and long).
int updateMul(bool doMul3, long x, Carry *carry, uint bits) {
  x = doMul3 ? x * 3 : x;
  x += *carry;
  int w = lowBits(x, bits);
  *carry = (x - w) >> bits;
  return w;
}

int update(long x, Carry *carry, uint bits) { return updateMul(false, x, carry, bits); }

// Simpler version of signbit(a).
uint signBit(double a) { return ((uint *)&a)[1] >> 31; }

uint oldBitlen(double a) { return EXP / NWORDS + signBit(a); }

// Reverse weighting, round, carry propagation for a pair of doubles; with optional MUL.
Word2 car0(bool doMul3, T2 u, Carry *carry, uint dummyPos, const T2 *iA, uint p) {
  T2 weight = iA[p];
  u *= fabs(weight);
  int a = updateMul(doMul3, toLong(u.x), carry, oldBitlen(weight.x));
  int b = updateMul(doMul3, toLong(u.y), carry, oldBitlen(weight.y));
  return (Word2) (a, b);
}

T2 weight(Word2 a, uint dummyPos, const T2 *A, uint p) { return U2(a.x, a.y) * fabs(A[p]); }

#endif


// Generic (the same for NTT/FP) code below.

// Carry propagation.
Word2 car1(Word2 a, Carry *carry, uint pos) {
  pos *= 2;
  a.x = update(a.x, carry, bitlen(pos + 0));
  a.y = update(a.y, carry, bitlen(pos + 1));
  return a;
}

T2 foo2(T2 a, T2 b) {
  a = addsub(a);
  b = addsub(b);
  return addsub(U2(mul1(a.x, b.x), mul1(a.y, b.y)));
}

// computes 2*[x^2+y^2 + i*(2*x*y)]. Needs a name.
T2 foo(T2 a) { return foo2(a, a); }

#define X2(a, b) { T2 t = a; a = add(t, b); b = sub(t, b); }
#define SWAP(a, b) { T2 t = a; a = b; b = t; }

void fft4Core(T2 *u) {
  X2(u[0], u[2]);
  X2(u[1], u[3]);
  u[3] = mul_t4(u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);
}

void fft8Core(T2 *u) {
  for (int i = 0; i < 4; ++i) { X2(u[i], u[i + 4]); }
  u[5] = mul_t8(u[5]);
  u[6] = mul_t4(u[6]);
  u[7] = mul_3t8(u[7]);
  
  fft4Core(u);
  fft4Core(u + 4);
}

void fft4(T2 *u) {
  fft4Core(u);
  SWAP(u[1], u[2]);
}

void fft8(T2 *u) {
  fft8Core(u);
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

void shufl(local T *lds, T2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  uint m = me / f;
  
  for (int b = 0; b < 2; ++b) {
    if (b) { bar(); }
    for (uint i = 0; i < n; ++i) { lds[(m + i * 256 / f) / n * f + m % n * 256 + me % f] = ((T *) (u + i))[b]; }
    bar();
    for (uint i = 0; i < n; ++i) { ((T *) (u + i))[b] = lds[i * 256 + me]; }
  }
  bar();
}

void tabMul(const G T2 *trig, T2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  for (int i = 1; i < n; ++i) { u[i] = mul(u[i], trig[me / f + i * (256 / f)]); }
}

void fft1kImpl(local T *lds, T2 *u, const G T2 *trig) {
  fft4(u);
  shufl(lds,   u, 4, 64);
  tabMul(trig, u, 4, 64);
  
  fft4(u);
  bar();
  shufl(lds,   u, 4, 16);
  tabMul(trig, u, 4, 16);
  
  fft4(u);
  bar();
  shufl(lds,   u, 4, 4);
  tabMul(trig, u, 4, 4);

  fft4(u);
  bar();
  shufl(lds,   u, 4, 1);
  tabMul(trig, u, 4, 1);

  fft4(u);
}

void fft2kImpl(local T *lds, T2 *u, const G T2 *trig) {
  fft8(u);
  shufl(lds,   u, 8, 32);
  tabMul(trig, u, 8, 32);

  fft8(u);
  bar();
  shufl(lds,   u, 8, 4);
  tabMul(trig, u, 8, 4);
  
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

  bar();
  for (int i = 1; i < 4; ++i) {
    u[i]     = mul(u[i],     trig[i * 512       + me]);
    u[i + 4] = mul(u[i + 4], trig[i * 512 + 256 + me]);
  }

  fft4(u);
  fft4(u + 4);

  // fix order: interleave u[0:4] and u[4:8], like (u.even, u.odd) = (u.lo, u.hi).
  SWAP(u[1], u[2]);
  SWAP(u[1], u[4]);
  SWAP(u[5], u[6]);
  SWAP(u[3], u[6]);
}

// choose between 1K and 2K based on N.
void fftImpl(uint N, local T *lds, T2 *u, const G T2 *trig) {
  if (N == 4) { fft1kImpl(lds, u, trig); } else { fft2kImpl(lds, u, trig); }
}

void read(uint N, T2 *u, G T2 *in, uint base) {
  for (int i = 0; i < N; ++i) { u[i] = in[base + i * 256 + (uint) get_local_id(0)]; }
}

void write(uint N, T2 *u, G T2 *out, uint base) {
  for (int i = 0; i < N; ++i) { out[base + i * 256 + (uint) get_local_id(0)] = u[i]; }
}

// FFT of size N * 256.
void fft(uint N, local T *lds, T2 *u, G T2 *io, const G T2 *trig) {
  uint g = get_group_id(0);
  uint step = g * (N * 256);
  io += step;

  read(N, u, io, 0);
  fftImpl(N, lds, u, trig);
  write(N, u, io, 0);
}

// fftPremul: weight words with "A" (for IBDWT) followed by FFT.
void fftPremul(uint N, uint H, local T *lds, T2 *u, const G Word2 *in, G T2 *out, const G T2 *A, const G T2 *trig) {
  uint g = get_group_id(0);
  uint step = N * 256 * g;
  in  += step;
  out += step;
  
  uint me = get_local_id(0);

  for (int i = 0; i < N; ++i) {
    uint pos = g + H * 256 * i + H * me;
    u[i] = weight(in[256 * i + me], pos, A, me + 256 * i + step);
  }

  fftImpl(N, lds, u, trig);

  write(N, u, out, 0);
}

void reverse8(local T2 *lds, T2 *u, bool bump) {
  uint me = get_local_id(0);
  uint rm = 255 - me + bump;
  
  bar();

  lds[rm + 0 * 256] = u[7];
  lds[rm + 1 * 256] = u[6];
  lds[rm + 2 * 256] = u[5];
  lds[bump ? ((rm + 3 * 256) & 1023) : (rm + 3 * 256)] = u[4];
  
  bar();
  for (int i = 0; i < 4; ++i) { u[4 + i] = lds[256 * i + me]; }
}

void reverse4(local T2 *lds, T2 *u, bool bump) {
  uint me = get_local_id(0);
  uint rm = 255 - me + bump;
  
  bar();

  lds[rm + 0 * 256] = u[3];
  lds[bump ? ((rm + 256) & 511) : (rm + 256)] = u[2];
  
  bar();
  u[2] = lds[me];
  u[3] = lds[me + 256];
}

void reverse(uint N, local T2 *lds, T2 *u, bool bump) {
  if (N == 4) {
    reverse4(lds, u, bump);
  } else {
    reverse8(lds, u, bump);
  }
}

#ifdef FFT_NTT

T2 weightAndCarry(T2 u, T *carry, uint pos, const T2 *dummyA, uint dummyP) {
  return car0(false, u, carry, pos, dummyA, dummyP);
}

// No carry out. The final carry is "absorbed" in the last word.
T2 carryAndWeightFinal(T2 u, T carry, uint pos, const T2 *dummyA, uint dummyP) {
  u.x = carryStep(u.x, &carry, bitlen(2 * pos + 0));
  u.y += carry;
  return weight(u, pos, dummyA, dummyP);
}

#else

// Applies inverse weight "iA" and rounding, and propagates carry over the two words.
T2 weightAndCarry(T2 u, T *carry, uint dummyPos, const T2 *iA, uint p) {
  T2 w = iA[p];
  u = rint(u * fabs(w)); // reverse weight and round.
  u.x = carryStep(u.x, carry, oldBitlen(w.x));
  u.y = carryStep(u.y, carry, oldBitlen(w.y));
  return u;
}

// No carry out. The final carry is "absorbed" in the last word.
T2 carryAndWeightFinal(T2 u, T carry, uint dummyPos, const T2 *A, uint p) {
  T2 w = A[p];
  u.x = carryStep(u.x, &carry, oldBitlen(w.x));
  u.y += carry;
  return u * fabs(w);
}

#endif

// The "carryConvolution" is equivalent to the sequence: fft, carryA, carryB, fftPremul.
// It uses "stareway" carry data forwarding from group K to group K+1.
// N gives the FFT size, W = N * 256.
// H gives the nuber of "lines" of FFT.
void carryConvolution(uint N, uint H, local T *lds, T2 *u,
                  G T2 *io, G T *carryShuttle, volatile global uint *ready,
                  const G T2 *A, const G T2 *iA, const G T2 *trig) {
  uint W = N * 256;

  uint gr = get_group_id(0);
  uint gm = gr % H;
  uint me = get_local_id(0);
  uint step = gm * W;

  io    += step;
  A     += step;
  iA    += step;

  read(N, u, io, 0);
  fftImpl(N, lds, u, trig);

  for (int i = 0; i < N; ++i) {
    uint p = i * 256 + me;
    uint pos = gm + H * 256 * i + H * me;
    T carry = 0;
    u[i] = weightAndCarry(conjugate(u[i]), &carry, pos, iA, p);
    if (gr < H) { carryShuttle[gr * W + p] = carry; }
  }

  bigBar();

  // Signal that this group is done writing the carry.
  if (gr < H && me == 0) { atomic_xchg(&ready[gr], 1); }

  if (gr == 0) { return; }

  // Wait until the previous group is ready with the carry.
  if (me == 0) { while(!atomic_xchg(&ready[gr - 1], 0)); }

  bigBar();
  
  for (int i = 0; i < N; ++i) {
    uint p = i * 256 + me;
    uint pos = gm + H * 256 * i + H * me;
    double carry = carryShuttle[(gr - 1) * W + ((p - gr / H) & (W - 1))];
    u[i] = carryAndWeightFinal(u[i], carry, pos, A, p);
  }

  fftImpl(N, lds, u, trig);
  write(N, u, io, 0);
}

// Carry propagation with optional MUL-3, over CARRY_LEN words.
// Input is conjugated and inverse-weighted.
void carryACore(uint N, uint H, bool doMul3, const G T2 *in, const G T2 *A, G Word2 *out, G Carry *carryOut) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);
  uint gx = g % N;
  uint gy = g / N;

  uint step = 256 * gx + N * 256 * CARRY_LEN * gy;
  in     += step;
  out    += step;

  Carry carry = 0;

  for (int i = 0; i < CARRY_LEN; ++i) {
    uint pos = CARRY_LEN * gy + H * 256 * gx  + H * me + i;
    uint p = me + i * N * 256;
    out[p] = car0(doMul3, conjugate(in[p]), &carry, pos, A, step + p);
  }
  carryOut[g * 256 + me] = carry;
}

// The second round of carry propagation (16 words), needed to "link the chain" after carryA.
void carryBCore(uint N, uint H, G Word2 *io, const G Carry *carryIn) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);
  uint gx = g % N;
  uint gy = g / N;
  
  uint step = 256 * gx + N * 256 * CARRY_LEN * gy;
  io += step;

  uint HB = H / CARRY_LEN;
  
  uint prev = (gy + HB * 256 * gx + HB * me - 1) & (HB * N * 256 - 1);
  uint prevLine = prev % HB;
  uint prevCol  = prev / HB;
  Carry carry = carryIn[N * 256 * prevLine + prevCol];
  
  for (int i = 0; i < CARRY_LEN; ++i) {
    uint pos = CARRY_LEN * gy + H * 256 * gx + H * me + i;
    uint p = me + i * N * 256;
    io[p] = car1(io[p], &carry, pos);
    if (!carry) { return; }
  }
}

// Inputs normal (non-conjugate); outputs conjugate.
void csquare(uint W, uint H, G T2 *io, const G T2 *bigTrig) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);

  if (g == 0 && me == 0) {
    io[0]     = shl(foo(conjugate(io[0])), 2);
    io[W / 2] = shl(sq(conjugate(io[W / 2])), 3);
    return;
  }
  
  uint line = g / (W / 512);
  uint posInLine = g % (W / 512) * 256 + me;
  uint k = line * W + posInLine;
  uint v = ((H - line) & (H - 1)) * W + (W - 1) - posInLine + ((line - 1) >> 31);
  
  T2 a = io[k];
  T2 b = conjugate(io[v]);
  T2 t = swap(mul(bigTrig[W * 2 + H / 2 + line], bigTrig[posInLine]));
  
  X2(a, b);
  b = mul(b, conjugate(t));
  X2(a, b);

  a = sq(a);
  b = sq(b);

  X2(a, b);
  b = mul(b,  t);
  X2(a, b);
  
  io[k] = conjugate(a);
  io[v] = b;
}

// Like csquare(), but for multiplication.
void cmul(uint W, uint H, G T2 *io, const G T2 *in, const G T2 *bigTrig) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);

  if (g == 0 && me == 0) {
    io[0]     = shl(foo2(conjugate(io[0]), conjugate(in[0])), 2);
    io[W / 2] = shl(conjugate(mul(io[W / 2], in[W / 2])), 3);
    return;
  }
  
  uint line = g / (W / 512);
  uint posInLine = g % (W / 512) * 256 + me;
  uint k = line * W + posInLine;
  uint v = ((H - line) & (H - 1)) * W + (W - 1) - posInLine + ((line - 1) >> 31);
  
  T2 a = io[k];
  T2 b = conjugate(io[v]);
  T2 t = swap(mul(bigTrig[W * 2 + H / 2 + line], bigTrig[posInLine]));
  
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
  b = mul(b,  t);
  X2(a, b);
  
  io[k] = conjugate(a);
  io[v] = b;
}

void transposeCore(local T *lds, T2 *u) {
  uint me = get_local_id(0);
  for (int b = 0; b < 2; ++b) {
    if (b) { bar(); }
    for (int i = 0; i < 16; ++i) {
      uint l = i * 4 + me / 64;
      uint c = me % 64;
      lds[l * 64 + (c + l) % 64] = ((T *)(u + i))[b];
    }
    bar();
    for (int i = 0; i < 16; ++i) {
      uint c = i * 4 + me / 64;
      uint l = me % 64;
      ((T *)(u + i))[b] = lds[l * 64 + (c + l) % 64];
    }
  }
}

// M == max(W, H)
void transpose(uint W, uint H, uint M, local T *lds, const G T2 *in, G T2 *out, const G T2 *bigTrig) {
  uint GW = W / 64, GH = H / 64;
  uint g = get_group_id(0), gx = g % GW, gy = g / GW;
  gy = (gy + gx) % GH;
  in   += gy * 64 * W + gx * 64;
  out  += gy * 64     + gx * 64 * H;
  uint me = get_local_id(0), mx = me % 64, my = me / 64;
  
  T2 u[16];
  for (int i = 0; i < 16; ++i) {
    uint p = (my + i * 4) * W + mx;
    u[i] = in[p];
  }

  transposeCore(lds, u);
  
  for (int i = 0; i < 16; ++i) {
    uint k = mul24(gy * 64 + mx, gx * 64 + my + (uint) i * 4);
    u[i] = mul(u[i], bigTrig[M * 2 + k % (W * H / (M * 2))]);
    u[i] = mul(u[i], bigTrig[k / (W * H / (M * 2))]);

    uint p = (my + i * 4) * H + mx;
    out[p] = u[i];
  }
}

void halfSq(uint N, T2 *u, T2 *v, T2 tt, const G T2 *bigTrig, bool special) {
  uint g = get_group_id(0);
  uint me = get_local_id(0);
  for (int i = 0; i < N / 2; ++i) {
    T2 a = u[i];
    T2 b = conjugate(v[N / 2 + i]);
    T2 t = swap(mul(tt, bigTrig[256 * i + me]));
    if (special && i == 0 && g == 0 && me == 0) {
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
    v[N / 2 + i] = b;
  }
}

void convolution(uint N, uint H, local T *lds, T2 *u, T2 *v, G T2 *io, const G T2 *trig, const G T2 *bigTrig) {
  uint W = N * 256;
  uint g = get_group_id(0);
  uint me = get_local_id(0);
  read(N, u, io, g * W);
  fftImpl(N, lds, u, trig);
  reverse(N, (local T2 *) lds, u, g == 0);

  uint line2 = g ? H - g : (H / 2);
  read(N, v, io, line2 * W);
  bar();
  fftImpl(N, lds, v, trig);
  reverse(N, (local T2 *) lds, v, false);
  
  if (g == 0) { for (int i = N / 2; i < N; ++i) { SWAP(u[i], v[i]); } }

  halfSq(N, u, v, bigTrig[W * 2 + (H / 2) + g],     bigTrig, true);
  
  halfSq(N, v, u, bigTrig[W * 2 + (H / 2) + line2], bigTrig, false);

  if (g == 0) { for (int i = N / 2; i < N; ++i) { SWAP(u[i], v[i]); } }

  reverse(N, (local T2 *) lds, u, g == 0);
  bar();
  fftImpl(N, lds, u, trig);
  write(N, u, io, g * W);
  
  reverse(N, (local T2 *) lds, v, false);
  bar();
  fftImpl(N, lds, v, trig);
  write(N, v, io, line2 * W);
}


#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

#define N_WIDTH  (WIDTH  / 256)
#define N_HEIGHT (HEIGHT / 256)

#ifndef ALT_RESTRICT

#define P(x) global x * restrict
#define CP(x) const P(x)
typedef CP(T2) Trig;

#else

#define P(x) global x *
#define CP(x) const P(x)
typedef CP(T2) restrict Trig;

#endif

KERNEL(256) fftW(P(T2) io, Trig smallTrig) {
  local T lds[WIDTH];
  T2 u[N_WIDTH];
  fft(N_WIDTH, lds, u, io, smallTrig);
}

KERNEL(256) fftH(P(T2) io, Trig smallTrig) {
  local T lds[HEIGHT];
  T2 u[N_HEIGHT];
  fft(N_HEIGHT, lds, u, io, smallTrig);
}

KERNEL(256) fftP(CP(Word2) in, P(T2) out, CP(T2) A, Trig smallTrig) {
  local T lds[WIDTH];
  T2 u[N_WIDTH];
  fftPremul(N_WIDTH, HEIGHT, lds, u, in, out, A, smallTrig);
}

KERNEL(256) carryA(CP(T2) in, CP(T2) A, P(Word2) out, P(Carry) carryOut) {
  carryACore(N_WIDTH, HEIGHT, false, in, A, out, carryOut);
}

KERNEL(256) carryM(CP(T2) in, CP(T2) A, P(Word2) out, P(Carry) carryOut) {
  carryACore(N_WIDTH, HEIGHT, true, in, A, out, carryOut);
}

KERNEL(256) carryB(P(Word2) io, CP(Carry) carryIn) {
  carryBCore(N_WIDTH, HEIGHT, io, carryIn);
}

KERNEL(256) square(P(T2) io, Trig bigTrig)  { csquare(HEIGHT, WIDTH, io, bigTrig); }

KERNEL(256) multiply(P(T2) io, CP(T2) in, Trig bigTrig)  { cmul(HEIGHT, WIDTH, io, in, bigTrig); }

KERNEL(256) carryConv(P(T2) io, P(T) carryShuttle, volatile P(uint) ready,
                      CP(T2) A, CP(T2) iA, Trig smallTrig) {
  local T lds[WIDTH];
  T2 u[N_WIDTH];
  carryConvolution(N_WIDTH, HEIGHT, lds, u, io, carryShuttle, ready, A, iA, smallTrig);
}

KERNEL(256) tail(P(T2) io, Trig smallTrig, Trig bigTrig) {
  local T lds[HEIGHT];
  T2 u[N_HEIGHT];
  T2 v[N_HEIGHT];
  convolution(N_HEIGHT, WIDTH, lds, u, v, io, smallTrig, bigTrig);
}

KERNEL(256) transposeW(CP(T2) in, P(T2) out, Trig bigTrig) {
  local T lds[4096];
  transpose(WIDTH, HEIGHT, max(WIDTH, HEIGHT), lds, in, out, bigTrig);
}

KERNEL(256) transposeH(CP(T2) in, P(T2) out, Trig bigTrig) {
  local T lds[4096];
  transpose(HEIGHT, WIDTH, max(WIDTH, HEIGHT), lds, in, out, bigTrig);
}

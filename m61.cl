// GpuOwl, an OpenCL Mersenne primality test
// FGT (Fast Galoais Transform)
// Copyright (C) Mihai Preda.

// The data is organized in pairs of words in a matrix WIDTH x HEIGHT.
// The pair (a, b) is sometimes interpreted as the complex value a + i*b.
// The order of words is column-major (i.e. transposed from the usual row-major matrix order).

// Expected defines: WIDTH, HEIGHT, EXP.
// Defines to select between "Fast Gallois Transform" using either M(31) or M(61),
// and floating point FFT in either double or single precision.
// One of: FGT_31, FGT_61, FP_DP (double precision), FP_SP (single precision).
// If FGT, also LOG_ROOT2 is expected, LOG_ROOT2 == log2(32 / (NWORDS % 31) % 31).

// FGT: a GF(P^2) (Galois Field) convolution, with P == M(31) == 2^31-1 a Mersenne prime.
// GF(P^2) means Gaussian integers ("complex integers") with the real/imaginary part modulo P.

typedef long T;
typedef long2 T2;
#define SIZEOF_T 8

typedef long Word;
typedef long2 Word2;
typedef long Carry;

#include "shared.cl"

#define MBITS 61
// ((1ll << MBITS) - 1)
#define M 0x1fffffffffffffff

#define SBITS(r, k) assert(((r) >> (k)) == 0 || ((r) >> (k)) == -1)
#define UBITS(r, k) assert(((r) >> (k)) == 0)

// extract the low MBITS of "a" with balanced sign
#if 0
long lowMBits(ulong b) {
  long a = b & M;
  assert(a >= 0);

  // if top bit is 1, flip sign.
  long ret = (a >> (MBITS - 1)) ? a - M : a;
  // long ret = testBit(a, MBITS - 1) ? a - M : a;

  SBITS(ret, 60);
  return ret;
}
#endif

ulong OVERLOAD reduce(ulong a) {
  ulong ret = (a & M) + (a >> MBITS);
  UBITS(ret, 62);
  return ret;
}

long OVERLOAD reduce(long a) {
  a = (a & M) + (a >> MBITS);
  if (a > M/2) { a -= M; }
  assert(a <= M/2);
  assert(a >= -(M/2));
}

T shl(T a, u32 k) {
  assert(k < MBITS, "shift amount (k) is reduced mod MBITS");
  a = ((a << k) & M) + (a >> (MBITS - k));
  return reduce(a);
}

long balance(ulong a) {
  a = (a & M) + (a >> MBITS);
  if (a > M/2) { a -= M; }
  assert(a <= M/2);
  assert(a >= -(M/2));
}

ulong unbalance(long a) {
  // ulong b = (a < 0) ? U64(a) - 8 : U64(a);
  ulong b = U64(a) + ((a >> 63) << 3);
  // ulong b = U64(a) + (as_int2(a).y >> 28);
  return reduce(b);
}

ulong U64(long a) { return a; }
long  I64(ulong a) { return a; }

u64 madu64(u32 a, u32 b, u64 c) {
#if HAS_ASM && 0
  u64 result;
  __asm("v_mad_u64_u32 %0, vcc, %1, %2, %3\n"
        : "=v"(result) : "v"(a), "v"(b), "v"(c) : "vcc");
  return result;
#else
  return a * U64(b) + c;
#endif
}

u64 mulu64(u32 a, u32 b) {
  return a * U64(b);
  // return umad64(a, b, 0);
}

i64 madi64(i32 a, i32 b, i64 c) {
#if HAS_ASM && 0
  i64 result;
  __asm("v_mad_i64_i32 %0, vcc, %1, %2, %3\n"
        : "=v"(result) : "v"(a), "v"(b), "v"(c) : "vcc");
  return result;
#else
  return a * (i64) b + c;
#endif
}

u32 U32(ulong a) { return a; }



long OVERLOAD sq(u32 a) {
  u32 a0 = a;
  u32 a1 = a >> 32;
  UBITS(a1, 31);

  u64 r0 = umul64(a0, a0);
  u64 r1 = umad64(a1 << 1, a0, r0 >> 32);
  u64 r2 = umad64(a1, a1, r1 >> 32);

  u32 r1l = U32(r1);
  u64 r1shifted = (U64(r1l & 0x1fffffff) << 32) | (r1l >> 29);

  r2 = ((r2 << 3) & M) + (r2 >> (MBITS - 3));

  u64 sum = r2 + r1shifted + U32(r0);
  long ret = balance(sum);
  SBITS(ret, 61);
  return ret;
}

long OVERLOAD sq(i64 a) { return sq(abs(a)); }

long OVERLOAD mul(ulong a, ulong b) {
  u32 a0 = a;
  u32 a1 = a >> 32;
  u32 b0 = b;
  u32 b1 = b >> 32;

  u64 r0 = umul64(a0, b0);
  u64 t  = umad64(a1, b0, r0 >> 32);
  u64 r1 = umad64(a0, b1, t);
  assert(r1 >= t);  // r1 didn't overflow
  u64 r2 = umad64(a1, b1, r1 >> 32);
  UBITS(r2, 61);  // r2 being only 61 bits allows us to safely shift it up 3 bits
  r2 = r2 << 3;
  // r2 = ((r2 << 3) & M) | (r2 >> 58);

  u32 r1l = U32(r1);
  u64 r1shl = (U64(r1l & 0x1fffffff) << 32) | (r1l >> 29);
  u64 sum = r2 + r1shl + U32(r0);
  assert(sum >= r2);  // sum didn't overflow
  return balance(sum);
}

long OVERLOAD mul(long a, ulong b) {
  u64 ua = abs(a);
  i64 p = mul(ua, b);
  return a < 0 ? -p : p;
}

long OVERLOAD mul(long a, long b) {
  // return (b >= 0) ? mul(a, U64(b)) : -mul(a, U64(-b));
  return mul(a, unbalance(b));
}

// Complex mul: a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x
long2 OVERLOAD mul(long2 a, ulong2 b) {
  // "b" is precomputed thus fully reduced, i.e. 61 bits.
  UBITS(b.x, 61);
  UBITS(b.y, 61);

  long c = mul(a.x, b.x + b.y);
  long d = mul(a.y + a.x, b.y);
  long e = mul(a.y - a.x, b.x);
  return (long2) (c - d, c + e);
}

long2 sq(long2 a) {
#if 1
  SBITS(a.x, 62); // to afford a.x<<1 below
  return (long2) (mul(a.x + a.y, a.x - a.y), mul(a.x << 1, a.y));
#elif 0
  ulong uy = unbalance(a.y);
  return (long2) (sq(a.x) - sq(uy), mul(a.x, uy << 1));
#else
  u64 x = abs(a.x);
  u64 y = abs(a.y);
  u64 p = mul(x, y << 1);
  return (long2) (mul(I64(x) - I64(y), x + y), a.x < 0 ^ a.y < 0 ? -p : p);
#endif
}

// mul with (0, 1). (twiddle of tau/4, sqrt(-1) aka "i").
long2 mul_t4(long2 a) { return (long2) (-a.y, a.x); }

// mul with (2^30, 2^30). (twiddle of tau/8)
long2 mul_t8(long2 a) { return (long2) (shl(a.x - a.y, 30), shl(a.x + a.y, 30)); }

// mul with (-2^30, 2^30). (twiddle of 3*tau/8)
long2 mul_3t8(long2 a) { return U2(-shl(a.x + a.y, 30), shl(a.x - a.y, 30)); }

// --- carry ---

typedef u32 Roundoff;

i64 convert(T a, Roundoff *roundoff) {
  a = reduce(a);
  assert(abs(a) <= M/2);
  // u32 slack = (M/2 - abs(a)) >> (MBITS - 32);
  u32 dist = abs(a) >> (MBITS - 32);
  *roundoff = max(*roundoff, dist);
  return a;
}

Carry unweight(T x, uint pos) {
  x = (x + ((x + 1) >> MBITS)) & M; // if x==M, set it to 0.
  return shl1(x, (extra(pos) * (MBITS - LOG_ROOT2) + (MBITS - LOG_NWORDS - 2)) % MBITS);
}

// Reverse weighting and carry propagation.
Word2 unweightAndCarry(uint mul, TT u, Carry *carry, uint pos, const G TT *dummyA, uint dummyP) {
  Word x = carryStep(mul * unweight(u.x, 2 * pos + 0), carry, bitlen(2 * pos + 0));
  Word y = carryStep(mul * unweight(u.y, 2 * pos + 1), carry, bitlen(2 * pos + 1));
  return (Word2) (x, y);
}

// NWORDS-th order root of 2: root2 ^ NWORDS == 2 (mod M31)
T weight1(Word x, uint pos) { return shl1(x, (extra(pos) * LOG_ROOT2) % MBITS); }

TT weightAux(Word2 a, uint pos) { return U2(weight1(a.x, 2 * pos + 0), weight1(a.y, 2 * pos + 1)); }

TT weight(Word2 a, uint pos, const G TT *dummyA, uint dummyP) { return weightAux(a, pos); }

// No carry out. The final carry is "absorbed" in the last word.
TT carryAndWeightFinal(Word2 u, Carry carry, uint pos, const G TT *dummyA, uint dummyP) {
  u.x = carryStep(u.x, &carry, bitlen(2 * pos + 0));
  u.y = u.y + carry;
  return weightAux(u, pos);
}


// Generic code below.

// Carry propagation from word and carry.
/*
Word2 carryWord(Word2 a, Carry *carry, uint pos) {
  a.x = carryStep(a.x, carry, bitlen(2 * pos + 0));
  a.y = carryStep(a.y, carry, bitlen(2 * pos + 1));
  return a;
}
*/

#include "shared_tail.cl"

void shufl(uint WG, local T *lds, TT *u, uint n, uint f) {
  uint me = get_local_id(0);
  uint m = me / f;
  
  for (int b = 0; b < 2; ++b) {
    if (b) { bar(); }
    for (uint i = 0; i < n; ++i) { lds[(m + i * WG / f) / n * f + m % n * WG + me % f] = ((T *) (u + i))[b]; }
    bar();
    for (uint i = 0; i < n; ++i) { ((T *) (u + i))[b] = lds[i * WG + me]; }
  }
  amd_fence();
}

void tabMul(uint WG, const G TT *trig, TT *u, uint n, uint f) {
  uint me = get_local_id(0);
  for (int i = 1; i < n; ++i) { u[i] = mul(u[i], trig[me / f + i * (WG / f)]); }
}

void fft1kImpl(local T *lds, TT *u, const G TT *trig) {
  for (int s = 6; s >= 0; s -= 2) {
    fft4(u);
    
    if (s != 6) { bar(); }
    shufl (256, lds,  u, 4, 1 << s);
    tabMul(256, trig, u, 4, 1 << s);
  }

  fft4(u);
}

// WG:512, LDS:32KB, u:8.
void fft4kImpl(local T *lds, TT *u, const G TT *trig) {
  for (int s = 6; s >= 0; s -= 3) {
    fft8(u);

    if (s != 6) { bar(); }
    shufl (512, lds,  u, 8, 1 << s);
    tabMul(512, trig, u, 8, 1 << s);
  }
  fft8(u);
}

// WG:256, LDS:16KB?, u:8
void fft2kImpl(local T *lds, TT *u, const G TT *trig) {
  for (int s = 5; s >= 2; s -= 3) {
      fft8(u);
      if (s != 5) { bar(); }
      shufl (256, lds,  u, 8, 1 << s);
      tabMul(256, trig, u, 8, 1 << s);
  }
  
  /*
  fft8(u);
  shufl(256, lds,   u, 8, 32);
  tabMul(256, trig, u, 8, 32);

  fft8(u);
  bar();
  shufl(256, lds,   u, 8, 4);
  tabMul(256, trig, u, 8, 4);
  */
  
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

  amd_fence();
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

// choose between 1K and 2K based on N.
void fftImpl(uint SIZE, local T *lds, TT *u, const G TT *trig) {
  if (SIZE == 1024) {
    fft1kImpl(lds, u, trig);
  } else if (SIZE == 2048) {
    fft2kImpl(lds, u, trig);
  } else if (SIZE == 4096) {
    fft4kImpl(lds, u, trig);
  }
}

void read(uint WG, uint N, TT *u, G TT *in, uint base) {
  for (int i = 0; i < N; ++i) { u[i] = in[base + i * WG + (uint) get_local_id(0)]; }
}

void write(uint WG, uint N, TT *u, G TT *out, uint base) {
  for (int i = 0; i < N; ++i) { out[base + i * WG + (uint) get_local_id(0)] = u[i]; }
}

// fftPremul: weight words with "A" (for IBDWT) followed by FFT.
void fftPremul(uint N, uint H, local T *lds, TT *u, const G Word2 *in, G TT *out, const G TT *A, const G TT *trig) {
  uint g = get_group_id(0);
  uint step = N * 256 * g;
  in  += step;
  out += step;
  A   += step;
  
  uint me = get_local_id(0);

  for (int i = 0; i < N; ++i) {
    uint pos = g + H * 256 * i + H * me;
    u[i] = weight(in[256 * i + me], pos, A, me + 256 * i);
  }

  fftImpl(N * 256, lds, u, trig);

  write(256, N, u, out, 0);
}

// Carry propagation with optional MUL-3, over CARRY_LEN words.
// Input is conjugated and inverse-weighted.
void carryACore(uint N, uint H, uint mul, const G TT *in, const G TT *A, G Word2 *out, G Carry *carryOut) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);
  uint gx = g % N;
  uint gy = g / N;

  uint step = 256 * gx + N * 256 * CARRY_LEN * gy;
  in  += step;
  out += step;
  A   += step;

  Carry carry = 0;

  for (int i = 0; i < CARRY_LEN; ++i) {
    uint pos = CARRY_LEN * gy + H * 256 * gx  + H * me + i;
    uint p = me + i * N * 256;
    out[p] = unweightAndCarry(mul, conjugate(in[p]), &carry, pos, A, p);
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
    io[p] = carryWord(io[p], &carry, pos);
    if (!carry) { return; }
  }
}

// Inputs normal (non-conjugate); outputs conjugate.
void csquare(uint W, uint H, G TT *io, const G TT *bigTrig) {
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
  
  TT a = io[k];
  TT b = conjugate(io[v]);
  TT t = swap(mul(bigTrig[W * 2 + H / 2 + line], bigTrig[posInLine]));
  
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
void cmul(uint W, uint H, G TT *io, const G TT *in, const G TT *bigTrig) {
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
  
  TT a = io[k];
  TT b = conjugate(io[v]);
  TT t = swap(mul(bigTrig[W * 2 + H / 2 + line], bigTrig[posInLine]));
  
  X2(a, b);
  b = mul(b, conjugate(t));
  X2(a, b);
  
  TT c = in[k];
  TT d = conjugate(in[v]);
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

void transposeCore(local T *lds, TT *u) {
  uint me = get_local_id(0);
  for (int b = 0; b < 2; ++b) {
    if (b) { bar(); }
    for (int i = 0; i < 16; ++i) {
      uint l = i * 4 + me / 64;
      // uint c = me % 64;
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

// M == max(W, H)
void transpose(uint W, uint H, uint MAX, local T *lds, const G TT *in, G TT *out, const G TT *bigTrig) {
  uint GW = W / 64, GH = H / 64;
  uint g = get_group_id(0), gx = g % GW, gy = g / GW;
  gy = (gy + gx) % GH;
  in   += gy * 64 * W + gx * 64;
  out  += gy * 64     + gx * 64 * H;
  uint me = get_local_id(0), mx = me % 64, my = me / 64;
  
  TT u[16];
  for (int i = 0; i < 16; ++i) {
    uint p = (my + i * 4) * W + mx;
    u[i] = in[p];
  }

  transposeCore(lds, u);
  
  for (int i = 0; i < 16; ++i) {
    uint k = mul24(gy * 64 + mx, gx * 64 + my + (uint) i * 4);
    u[i] = mul(u[i], bigTrig[MAX * 2 + k % (W * H / (MAX * 2))]);
    u[i] = mul(u[i], bigTrig[k / (W * H / (MAX * 2))]);

    uint p = (my + i * 4) * H + mx;
    out[p] = u[i];
  }
}

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

#define N_WIDTH  (WIDTH  / 256)
#define N_HEIGHT (HEIGHT / 256)

#ifndef ALT_RESTRICT

#define P(x) global x * restrict
#define CP(x) const P(x)
typedef CP(TT) Trig;

#else

#define P(x) global x *
#define CP(x) const P(x)
typedef CP(TT) restrict Trig;

#endif


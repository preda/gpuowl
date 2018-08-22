// gpuOwl, an OpenCL Mersenne primality test.
// Copyright (C) 2017-2018 Mihai Preda.

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

#include "shared.h"

// Expected defines: EXP the exponent.
// WIDTH, SMALL_HEIGHT, MIDDLE.

#define BIG_HEIGHT (SMALL_HEIGHT * MIDDLE)
#define ND (WIDTH * BIG_HEIGHT)
#define NWORDS (ND * 2u)

#if WIDTH == 1024
#define NW 4
#else
#define NW 8
#endif

#if SMALL_HEIGHT == 1024
#define NH 4
#else
#define NH 8
#endif

#define G_W (WIDTH / NW)
#define G_H (SMALL_HEIGHT / NH)

// Used in bitlen() and weighting.
#define STEP (NWORDS - (EXP % NWORDS))

uint extra(uint k) { return ((ulong) STEP) * k % NWORDS; }

// Is the word at pos a big word (BASE_BITLEN+1 bits)? (vs. a small, BASE_BITLEN bits word).
bool isBigWord(uint k) { return extra(k) + STEP < NWORDS; }
// { return extra(k) < extra(k + 1); }

// Number of bits for the word at pos.
uint bitlen(uint k) { return EXP / NWORDS + isBigWord(k); }

// Propagate carry this many pairs of words.
#define CARRY_LEN 16

typedef double T;
typedef double2 T2;
typedef int Word;
typedef int2 Word2;
typedef long Carry;

T2 U2(T a, T b) { return (T2)(a, b); }

// complex mul
T2 mul(T2 a, T2 b) { return U2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

// complex square
T2 sq(T2 a) { return U2((a.x + a.y) * (a.x - a.y), 2 * a.x * a.y); }

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

// Simpler version of signbit(a).
uint signBit(double a) { return ((uint *)&a)[1] >> 31; }

uint oldBitlen(double a) { return EXP / NWORDS + signBit(a); }

Carry unweight(T x, T weight) { return rint(x * fabs(weight)); }  

Word2 unweightAndCarry(uint mul, T2 u, Carry *carry, T2 weight) {
  Word a = carryStep(mul * unweight(u.x, weight.x), carry, oldBitlen(weight.x));
  Word b = carryStep(mul * unweight(u.y, weight.y), carry, oldBitlen(weight.y));
  return (Word2) (a, b);
}

T2 weightAux(Word x, Word y, T2 weight) { return U2(x, y) * fabs(weight); }

T2 weight(Word2 a, T2 w) { return weightAux(a.x, a.y, w); }

// No carry out. The final carry is "absorbed" in the last word.
T2 carryAndWeightFinal(Word2 u, Carry carry, T2 w) {
  Word x = carryStep(u.x, &carry, oldBitlen(w.x));
  Word y = u.y + carry;
  return weightAux(x, y, w);
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

#define X2(a, b) { T2 t = a; a = t + b; b = t - b; }
#define SWAP(a, b) { T2 t = a; a = b; b = t; }

void fft4Core(T2 *u) {
  X2(u[0], u[2]);
  X2(u[1], u[3]);
  u[3] = mul_t4(u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);
}

void fft4(T2 *u) {
  fft4Core(u);
  // revbin [0, 2, 1, 3] undo
  SWAP(u[1], u[2]);
}

void fft8Core(T2 *u) {
  for (int i = 0; i < 4; ++i) { X2(u[i], u[i + 4]); }
  u[5] = mul_t8(u[5]);
  u[6] = mul_t4(u[6]);
  u[7] = mul_3t8(u[7]);
  
  fft4Core(u);
  fft4Core(u + 4);
}

void fft8(T2 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

// Adapted from: Nussbaumer, "Fast Fourier Transform and Convolution Algorithms", 5.5.4 "5-Point DFT".
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

void shufl(uint WG, local T *lds, T2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  uint m = me / f;
  
  for (int b = 0; b < 2; ++b) {
    if (b) { bar(); }
    for (uint i = 0; i < n; ++i) { lds[(m + i * WG / f) / n * f + m % n * WG + me % f] = ((T *) (u + i))[b]; }
    bar();
    for (uint i = 0; i < n; ++i) { ((T *) (u + i))[b] = lds[i * WG + me]; }
  }
}

void tabMul(uint WG, const T2 *trig, T2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  for (int i = 1; i < n; ++i) { u[i] = mul(u[i], trig[me / f + i * (WG / f)]); }
}

// 64x8
void fft512(local T *lds, T2 *u, const T2 *trig) {
  for (int s = 3; s >= 0; s -= 3) {
    fft8(u);
    bar();
    shufl( 64,  lds, u, 8, 1 << s);
    tabMul(64, trig, u, 8, 1 << s);
  }
  fft8(u);
}

// 256x4
void fft1K(local T *lds, T2 *u, const T2 *trig) {
  for (int s = 6; s >= 0; s -= 2) {
    fft4(u);
    bar();
    shufl(256,   lds, u, 4, 1 << s);
    tabMul(256, trig, u, 4, 1 << s);
  }

  fft4(u);
}

// 512x8
void fft4K(local T *lds, T2 *u, const T2 *trig) {
  for (int s = 6; s >= 0; s -= 3) {
    fft8(u);
    if (s != 6) { bar(); }
    shufl( 512,  lds, u, 8, 1 << s);
    tabMul(512, trig, u, 8, 1 << s); 
  }

  fft8(u);
}

// 256x8
void fft2K(local T *lds, T2 *u, const T2 *trig) {
  for (int s = 5; s >= 2; s -= 3) {
    fft8(u);
    bar();
    shufl(256,   lds, u, 8, 1 << s);
    tabMul(256, trig, u, 8, 1 << s);
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

void read(uint WG, uint N, T2 *u, const T2 *in, uint base) {
  for (int i = 0; i < N; ++i) { u[i] = in[base + i * WG + (uint) get_local_id(0)]; }
}

void write(uint WG, uint N, T2 *u, T2 *out, uint base) {
  for (int i = 0; i < N; ++i) { out[base + i * WG + (uint) get_local_id(0)] = u[i]; }
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
#if   WIDTH ==  512
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
#if SMALL_HEIGHT == 512
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

KERNEL(256) fftMiddleIn(P(T2) io) {
  T2 u[MIDDLE];
  uint N = SMALL_HEIGHT / 256;
  uint g = get_group_id(0);
  uint gx = g % N;
  uint gy = g / N;
  uint me = get_local_id(0);
  io += BIG_HEIGHT * gy + 256 * gx;
  read(SMALL_HEIGHT, MIDDLE, u, io, 0);
  
#if MIDDLE == 5
  fft5(u);
#elif MIDDLE == 9
  fft9(u);
#elif MIDDLE != 1
#error
#endif
    
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

#if MIDDLE == 5
  fft5(u);
#elif MIDDLE == 9
  fft9(u);
#elif MIDDLE != 1
#error
#endif

  write(SMALL_HEIGHT, MIDDLE, u, io, 0);
}

// Carry propagation with optional MUL-3, over CARRY_LEN words.
// Input is conjugated and inverse-weighted.
void carryACore(uint mul, const T2 *in, const T2 *A, Word2 *out, Carry *carryOut) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);
  uint gx = g % NW;
  uint gy = g / NW;

  Carry carry = 0;

  for (int i = 0; i < CARRY_LEN; ++i) {
    uint p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;
    out[p] = unweightAndCarry(mul, conjugate(in[p]), &carry, A[p]);
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
    wu[i] = unweightAndCarry(1, conjugate(u[i]), &carry, iA[p]);
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
    u[i] = carryAndWeightFinal(wu[i], carry, A[p]);
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

KERNEL(G_H) mulFused(P(T2) io, CP(T2) in, Trig smallTrig) {
  local T lds[SMALL_HEIGHT];
  T2 u[NH], v[NH];
  T2 p[NH], q[NH];

  uint W = SMALL_HEIGHT;
  uint H = ND / W;

  uint line1 = get_group_id(0);
  uint line2 = line1 ? H - line1 : (H / 2);
  uint g1 = transPos(line1, MIDDLE, WIDTH);
  uint g2 = transPos(line2, MIDDLE, WIDTH);
  
  read(G_H, NH, u, io, g1 * SMALL_HEIGHT);
  read(G_H, NH, p, in, g1 * SMALL_HEIGHT);
  read(G_H, NH, v, io, g2 * SMALL_HEIGHT);
  read(G_H, NH, q, in, g2 * SMALL_HEIGHT);

  fft_HEIGHT(lds, u, smallTrig);
  fft_HEIGHT(lds, p, smallTrig);
  fft_HEIGHT(lds, v, smallTrig);
  fft_HEIGHT(lds, q, smallTrig);

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

KERNEL(G_H) tailFused(P(T2) io, Trig smallTrig) {
  local T lds[SMALL_HEIGHT];
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

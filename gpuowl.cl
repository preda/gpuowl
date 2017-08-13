// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void
#define CONST const global
#define SMALL_CONST constant

#define X1(a, b) { double  t = a; a = t + b; b = t - b; }
#define X2(a, b) { double2 t = a; a = t + b; b = t - b; }
#define S2(a, b) { double2 t = a; a = b; b = t; }

void bar() { barrier(CLK_LOCAL_MEM_FENCE); }
double2 muld(double2 u, double a, double b) { return (double2) { u.x * a - u.y * b, u.x * b + u.y * a}; }
double2 mul(double2 u, double2 v) { return muld(u, v.x, v.y); }

#define MUL(x, a, b) x = muld(x, a, b)
#define MUL_2(x, a, b) x = muld(x, a, b) * M_SQRT1_2
#define M(x, t) x = mul(x, t)

double2 sq(double2 u) {
  double t = u.x * u.y;
  X1(u.x, u.y);
  return (double2) (u.x * u.y, t + t);
}

double2 conjugate(double2 u) { return (double2)(u.x, -u.y); }

void fft4Core(double2 *u) {
  X2(u[0], u[2]);
  X2(u[1], u[3]);
  MUL(u[3], 0, -1);
  X2(u[0], u[1]);
  X2(u[2], u[3]);
}

void fft8Core(double2 *u) {
  for (int i = 0; i < 4; ++i) { X2(u[i], u[i + 4]); }
  MUL(u[6], 0, -1);
  MUL_2(u[5],  1, -1);
  MUL_2(u[7], -1, -1);
  
  fft4Core(u);
  fft4Core(u + 4);
}

void fft4(double2 *u) {
  fft4Core(u);
  S2(u[1], u[2]);
}

void fft8(double2 *u) {
  fft8Core(u);
  S2(u[1], u[4]);
  S2(u[3], u[6]);
}

void shufl(local double *lds, double2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  uint m = me / f;
  
  for (int b = 0; b < 2; ++b) {
    if (b) { bar(); }
    for (uint i = 0; i < n; ++i) { lds[(m + i * 256 / f) / n * f + m % n * 256 + me % f] = ((double *) (u + i))[b]; }
    bar();
    for (uint i = 0; i < n; ++i) { ((double *) (u + i))[b] = lds[i * 256 + me]; }
  }
}

void shuflBig(local double2 *lds, double2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  uint m = me / f;
  for (uint i = 0; i < n; ++i) { lds[(m + i * 256 / f) / n * f + m % n * 256 + me % f] = u[i]; }
  bar();
  for (uint i = 0; i < n; ++i) { u[i] = lds[i * 256 + me]; }
}

#define SWAP(a, b) { uint t = a; a = b; b = t; }

// returns e ^ (- 2 * pi * i / 2048)
// len(cosTab) == 513.
double2 cosSin2K(local double *tab, uint i) {
  uint pos  = i & 511;
  bool b0 = i & 512;
  bool b1 = i >> 10;  // i & 1024;
  uint p1 = b0 ? 512 - pos : pos;
  uint p2 = b0 ? pos : 512 - pos;
  double a = tab[p1];
  double b = tab[p2];
  return (double2) ((b0 == b1) ? a : -a, b1 ? b : -b);
  /*
  double a = (b0 == b1) ? tab[p1] : -tab[p1];
  double b = b1 ? tab[p2] : -tab[p2];
  return (double2) (a, b);
  */
}
/* 
  if (b0) {
    double a = b1 ? cosTab[pos2] : -cosTab[pos2];
    double b = b1 ? cosTab[pos] : -cosTab[pos];
  } else {
    double a = b1 ? -cosTab[pos] : cosTab[pos];
    double b = b1 ? cosTab[pos2] : -cosTab[pos2];
  }
  if (b0) { SWAP(pos, pos2); }
  double a = cosTab[pos];
  double b = cosTab[pos2];
  if (b0 != b1) { a = -a; }
  if (b1) { b = -b; }
  uint b0  = (i >> 9) & 1;
  uint b1  = (i >> 10) & 1;
}
*/

void tabMulNew(local double *trig, double2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  #pragma unroll 1
  for (int i = 1; i < n; ++i) { M(u[i], cosSin2K(trig, i * (me / f * f))); }
}

void tabMulOld(SMALL_CONST double2 *trig, double2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  for (int i = 1; i < n; ++i) { M(u[i], trig[me / f + i * (256 / f)]); }
}

/*
void tabMul2(SMALL_CONST double2 *trig, double2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  // n == 4
  uint p = me & ~(f - 1);
  M(u[1], trig[p]);
  M(u[2], trig[p + 256]);
  M(u[3], trig[p + 512]);
}
*/

/*
void shuffleMul(local double *trig, local double *lds, double2 *u, uint n, uint f) {
  bar();
  shufl(lds,   u, n, f);
  tabMul(trig, u, n, f);
}
*/

void shuffleMulBig(SMALL_CONST double2 *trig, local double2 *lds, double2 *u, uint n, uint f) {
  bar();
  shuflBig(lds, u, n, f);
  tabMulOld(trig,  u, n, f);
}

void fft1kBigLDS(local double2 *lds, double2 *u, SMALL_CONST double2 *trig1k) {
  fft4(u);
  shuflBig(lds,  u, 4, 64);
  tabMulOld(trig1k, u, 4, 64);
  
  fft4(u);
  shuffleMulBig(trig1k, lds, u, 4, 16);
  
  fft4(u);
  shuffleMulBig(trig1k, lds, u, 4, 4);

  fft4(u);
  shuffleMulBig(trig1k, lds, u, 4, 1);

  fft4(u);
}

void fft1kImpl(local double *lds, double2 *u, SMALL_CONST double2 *trig) {
  fft4(u);
  shufl(lds,      u, 4, 64);
  tabMulOld(trig, u, 4, 64);
  
  fft4(u);
  bar();
  shufl(lds,      u, 4, 16);
  tabMulOld(trig, u, 4, 16);
  
  fft4(u);
  bar();
  shufl(lds,      u, 4, 4);
  tabMulOld(trig, u, 4, 4);

  fft4(u);
  bar();
  shufl(lds,      u, 4, 1);
  tabMulOld(trig, u, 4, 1);

  fft4(u);
}

void fft2kImpl(local double *lds, double2 *u, SMALL_CONST double2 *trig) {
  fft8(u);
  shufl(lds,      u, 8, 32);
  tabMulOld(trig, u, 8, 32);

  fft8(u);
  bar();
  shufl(lds,      u, 8, 4);
  tabMulOld(trig, u, 8, 4);
  
  fft8(u);

  uint me = get_local_id(0);
  for (int b = 0; b < 2; ++b) {
    bar();
    for (int i = 0; i < 8; ++i) { lds[(me + i * 256) / 4 + me % 4 * 512] = ((double *) (u + i))[b]; }
    bar();
    for (int i = 0; i < 4; ++i) {
      ((double *) (u + i))[b]     = lds[i * 512 + me];
      ((double *) (u + i + 4))[b] = lds[i * 512 + me + 256];
    }
  }

  for (int i = 1; i < 4; ++i) {
    M(u[i],     trig[i * 512 + me]);
    M(u[i + 4], trig[i * 512 + me + 256]);
    // M(u[i],     cosSin2K(trig, i * me));
    // M(u[i + 4], cosSin2K(trig, i * (me + 256)));
  }
     
  fft4(u);
  fft4(u + 4);
}

// The fftPremul1K kernel merges two steps:
// 1. Premultiply integer words with "A" (for IBDWT)
// 2. FFT 1K.
KERNEL(256) fftPremul1K(CONST int2 *in, global double2 *out, CONST double2 *A, SMALL_CONST double2 *trig1k) {
  uint g = get_group_id(0);
  uint step = g * 1024;
  in  += step;
  A   += step;
  out += step;
  
  uint me = get_local_id(0);
  double2 u[4];

  for (int i = 0; i < 4; ++i) {
    int2 r = in[me + i * 256];
    u[i] = (double2)(r.x, r.y) * fabs(A[me + i * 256]);
  }

  local double lds[1024];
  fft1kImpl(lds, u, trig1k);

  for (int i = 0; i < 4; ++i) { out[me + i * 256] = u[i]; }  
}

KERNEL(256) fft1K(global double2 *io, SMALL_CONST double2 *trig1k) {
  uint g = get_group_id(0);
  uint step = g * 1024;
  io += step;

  uint me = get_local_id(0);
  double2 u[4];

  for (int i = 0; i < 4; ++i) { u[i] = io[i * 256 + me]; }

  local double lds[1024];
  fft1kImpl(lds, u, trig1k);
  
  for (int i = 0; i < 4; ++i) { io[i * 256 + me] = u[i]; }
}

double roundWithErr(double x, float *maxErr) {
  double rx = rint(x);
  *maxErr = max(*maxErr, fabs((float) (x - rx)));
  return rx;
}

long toLong(double x, float *maxErr) { return roundWithErr(x, maxErr); }

int lowBits(int u, uint bits) { return (u << (32 - bits)) >> (32 - bits); }

int updateA(long *carry, long x, uint bits) {
  long u = *carry + x;
  int w = lowBits(u, bits);
  *carry = (u - w) >> bits;
  return w;
}

int updateB(long *carry, long x, uint bits) {
  long u = *carry + x;
  int w = lowBits(((int) u) - 1, bits) + 1;
  *carry = (u - w) >> bits;
  return w;
}

// Simpler version of (a < 0).
uint signBit(double a) { return ((uint *)&a)[1] >> 31; }

uint bitlen(uint base, double a) { return base + signBit(a); }

int2 car0(long *carry, double2 u, double2 ia, float *maxErr, uint baseBits) {
  u *= fabs(ia);
  int a = updateA(carry, toLong(u.x, maxErr), bitlen(baseBits, ia.x));
  int b = updateB(carry, toLong(u.y, maxErr), bitlen(baseBits, ia.y));
  return (int2) (a, b);
}

int2 car1(long *carry, int2 r, uchar2 bits) {
  int a = updateA(carry, r.x, bits.x);
  int b = updateB(carry, r.y, bits.y);
  return (int2) (a, b);
}

double2 updateD(double x, int bits) {
  double carry = rint(ldexp(x, -bits));
  return (double2)(x - ldexp(carry, bits), carry);
}

double2 dar0(double *carry, double2 u, double2 ia, float *maxErr, uint baseBits) {
  u *= fabs(ia);
  double2 r0 = updateD(*carry + roundWithErr(u.x, maxErr), bitlen(baseBits, ia.x));
  double2 r1 = updateD(r0.y   + roundWithErr(u.y, maxErr), bitlen(baseBits, ia.y));
  *carry = r1.y;
  return (double2)(r0.x, r1.x);
}

double2 dar2(double carry, double2 u, double2 a, uint baseBits) {
  double2 r = updateD(carry + u.x, bitlen(baseBits, a.x));
  return (double2) (r.x, r.y + u.y) * fabs(a);
}

// The "amalgamation" kernel is equivalent to the sequence: fft1K, carryA, carryB, fftPremul1K.
void amalgamation1k(local double2 *lds, const uint baseBitlen,
                    global double2 *io, volatile global double *carry, volatile global uint *ready,
                    volatile global uint *globalErr,
                    CONST double2 *A, CONST double2 *iA, SMALL_CONST double2 *trig1k) {
  uint gr = get_group_id(0);
  uint gm = gr % 2048;
  uint me = get_local_id(0);
  uint step = gm * 1024;
  
  io    += step;
  A     += step;
  iA    += step;
  carry += ((int)gr - 1) * 1024;

  double2 u[4];
  for (int i = 0; i < 4; ++i) { u[i] = io[i * 256 + me]; }

  fft1kBigLDS(lds, u, trig1k);
  
  float err = 0;
  double2 r[4];

  // Fight the LLVM OpenCL compiler who doesn't care about # of VGPRs used.
  #pragma unroll 1
  for (int i = 0; i < 4; ++i) {
    uint p = i * 256 + me;
    double c = 0;
    r[i] = dar0(&c, conjugate(u[i]), iA[p], &err, baseBitlen);
    if (gr < 2048) { carry[1024 + p] = c; }
  }

  if (gm == 0 && me == 0) { r[0].x -= 2; }

#ifndef NO_ERR  
  local uint *maxErr = (local uint *) lds;
  if (me == 0) { *maxErr = 0; }
#endif
  
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  if (gr < 2048 && me == 0) { atomic_xchg(&ready[gr], 1); }
  if (gr == 0) { return; }

#ifndef NO_ERR
  atomic_max(maxErr, *(uint *)&err);
#endif
  
  if (me == 0) { while(!atomic_xchg(&ready[gr - 1], 0)); }
  bar();

#ifndef NO_ERR
  if (me == 0) { atomic_max(globalErr, *maxErr); }
#endif
  
  for (int i = 0; i < 4; ++i) {
    uint p = i * 256 + me;
    u[i] = dar2(carry[(p - gr / 2048) & 1023], r[i], A[p], baseBitlen);
  }

  fft1kBigLDS(lds, u, trig1k);
  
  for (int i = 0; i < 4; ++i) { io[i * 256 + me]  = u[i]; }
}

KERNEL(256) mega(const uint baseBitlen,
                 global double2 *io, volatile global double *carry, volatile global uint *ready,
                 volatile global uint *globalErr,
                 CONST double2 *A, CONST double2 *iA, SMALL_CONST double2 *trig1k) {
  local double2 lds[1024];
  amalgamation1k(lds, baseBitlen, io, carry, ready, globalErr, A, iA, trig1k);
}

// computes 8*[x^2+y^2 + i*(2*x*y)]. Needs a name.
double2 foo(double2 a) {
  /*
    double t = a.x * a.y;
    a *= a;
    return (double2)((a.x + a.y) * 8, t * 16);
  */
  double t = a.x * a.y * 2;
  double s = a.x + a.y;
  return 8 * (double2)(s * s - t, t);
}

void reverse1(local double2 *lds, double2 *u, bool bump) {
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

void reverse2(local double2 *lds, double2 *u, bool bump) {
  uint me = get_local_id(0);
  uint rm = 255 - me + bump;
  
  bar();
  lds[rm + 0 * 256] = u[7];
  lds[rm + 1 * 256] = u[3];
  lds[rm + 2 * 256] = u[6];
  lds[bump ? ((rm + 3 * 256) & 1023) : (rm + 3 * 256)] = u[2];
  u[2] = u[1];
  u[1] = u[4];
  u[3] = u[5];

  bar();
  for (int i = 0; i < 4; ++i) { u[4 + i] = lds[256 * i + me]; }
}

/*
KERNEL(256) test(global double2 *out, global double *cos2k) {
  local trig[513];
  
  for (int i = 0; i < 8; ++i) {
    
  }
  uint g = get_group_id(0);
  uint me = get_local_id(0);
  // out[get_global_id(0)] = -in[get_global_id(0)];
  // local double lds[2048];
  double2 u[8];
  for (int i = 0; i < 8; ++i) { u[i] = in[g * 2048 + i * 256 + me]; }
  fft8(u);
  for (int i = 0; i < 8; ++i) { out[g * 2048 + i * 256 + me] = u[i]; }  
  // fft2kImpl(lds, u, trig2k);
}
*/

KERNEL(256) tail(global double2 *io, SMALL_CONST double2 *trig, CONST double2 *bigTrig) {
  uint g = get_group_id(0);
  uint me = get_local_id(0);
  local double lds[2048];
  // local double trig[513];
  
  double2 u[8];  
  for (int i = 0; i < 8; ++i) { u[i] = io[g * 2048 + i * 256 + me]; }
  // trig[me]       = cos2k[me];
  // trig[me + 256] = cos2k[me + 256];
  // if (me == 0) { trig[512] = 0; }
  fft2kImpl(lds, u, trig);

  reverse2((local double2 *) lds, u, g == 0);

  double2 v[8];
  uint line2 = g ? 1024 - g : 512;
  for (int i = 0; i < 8; ++i) { v[i] = io[line2 * 2048 + i * 256 + me]; }
  bar(); fft2kImpl(lds, v, trig);

  reverse2((local double2 *) lds, v, false);
  
  if (g == 0) { for (int i = 0; i < 4; ++i) { S2(u[4 + i], v[4 + i]); } }
  
  for (int i = 0; i < 4; ++i) {
    double2 a = u[i];
    double2 b = conjugate(v[4 + i]);
    double2 t = bigTrig[g * 1024 + 256 * i + me];
    if (i == 0 && g == 0 && me == 0) {
      a = foo(a);
      b = 8 * sq(b);
    } else {
      X2(a, b);
      M(b, conjugate(t));
      X2(a, b);
      a = sq(a);
      b = sq(b);
      X2(a, b);
      M(b,  t);
      X2(a, b);
    }
    u[i]     = conjugate(a);
    v[4 + i] = b;
  }

  for (int i = 0; i < 4; ++i) {
    double2 a = v[i];
    double2 b = conjugate(u[4 + i]);
    double2 t = bigTrig[line2 * 1024 + 256 * i + me];
    X2(a, b);
    M(b, conjugate(t));
    X2(a, b);
    a = sq(a);
    b = sq(b);
    X2(a, b);
    M(b,  t);
    X2(a, b);
    v[i]     = conjugate(a);
    u[4 + i] = b;
  }

  if (g == 0) { for (int i = 0; i < 4; ++i) { S2(u[4 + i], v[4 + i]); } }

  reverse1((local double2 *) lds, u, g == 0);
  bar(); fft2kImpl(lds, u, trig);
  for (int i = 0; i < 4; ++i) {
    io[g * 2048 + i * 512 + me]       = u[i];
    io[g * 2048 + i * 512 + 256 + me] = u[i + 4];
  }
  
  reverse1((local double2 *) lds, v, false);
  bar(); fft2kImpl(lds, v, trig);
  for (int i = 0; i < 4; ++i) {
    io[line2 * 2048 + i * 512 + me]       = v[i];
    io[line2 * 2048 + i * 512 + 256 + me] = v[i + 4];
  }
}

/*
KERNEL(256) fft2K(global double2 *io, SMALL_CONST double2 *trig2k) {
  uint g = get_group_id(0);
  io += g * 2048;
  
  uint me = get_local_id(0);
  double2 u[8];

  for (int i = 0; i < 8; ++i) { u[i] = io[me + i * 256]; }

  local double lds[2048];
  fft2kImpl(lds, u, trig2k);

  for (int i = 0; i < 4; ++i) {
    io[me + i * 512]       = u[i];
    io[me + i * 512 + 256] = u[i + 4];
  }  
}

KERNEL(256) fft2K_1K(CONST double2 *in, global double2 *out, SMALL_CONST double2 *trig2k) {
  uint g = get_group_id(0), gx = g % 16, gy = g / 16, lg = gy + gx * 64;
  in  += g * 64;
  out += lg * 2048;
  
  uint me = get_local_id(0), mx = me % 64, my = me / 64;
  double2 u[8];

  for (int i = 0; i < 8; ++i) { u[i] = in[mx + (i * 4 + my) * 64 * 1024]; }

  local double lds[2048];
  fft2kImpl(lds, u, trig2k);

  for (int i = 0; i < 4; ++i) {
    out[me + i * 512]       = u[i];
    out[me + i * 512 + 256] = u[i + 4];
  }
}
*/

// conjugates input
KERNEL(256) carryA(const uint baseBits,
                   CONST double2 *in, CONST double2 *A, global int2 *out,
                   global long *carryOut, global uint *globalMaxErr) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);

  uint step = g % 4 * 256 + g / 4 * 8 * 1024;
  in     += step;
  A      += step;
  out    += step;

  float maxErr = 0;
  long carry   = (g == 0 && me == 0) ? -2 : 0;

  for (int i = 0; i < 8; ++i) {    
    uint p = me + i * 1024;
    out[p] = car0(&carry, conjugate(in[p]), A[p], &maxErr, baseBits);
  }

  carryOut[me + g * 256] = carry;

  local uint localMaxErr;
  if (me == 0) { localMaxErr = 0; }
  bar();
  atomic_max(&localMaxErr, *(uint *)&maxErr);
  bar();
  if (me == 0) { atomic_max(globalMaxErr, localMaxErr); }
}

void carryBCore(uint H, global int2 *in, CONST long *carryIn, CONST uchar2 *bitlen) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);
  
  uint step = g % 4 * 256 + g / 4 * 8 * 1024;
  in     += step;
  bitlen += step;
  
  uint prev = (g / 4 + (g % 4 * 256 + me) * (H / 8) - 1) & ((H / 8) * 1024 - 1);
  uint line = prev % (H / 8);
  uint col  = prev / (H / 8);
  long carry = carryIn[line * 1024 + col];
  
  for (int i = 0; i < 8; ++i) {
    uint p = me + i * 1024;
    in[p] = car1(&carry, in[p], bitlen[p]);
    if (!carry) { return; }
  }
}

KERNEL(256) carryB_2K(global int2 *in, global long *carryIn, CONST uchar2 *bitlen) {
  carryBCore(2048, in, carryIn, bitlen);
}

/*
// Inputs normal (non-conjugate); outputs conjugate.
void csquare(uint W, global double2 *io, CONST double2 *trig) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);

  if (g == 0 && me == 0) {
    io[0]    = foo(conjugate(io[0]));
    io[1024] = 8 * sq(conjugate(io[1024]));
    return;
  }
  
  uint line = g / (W / 512);
  uint posInLine = g % (W / 512) * 256 + me;
  uint k = line * W + posInLine;
  uint v = ((1024 - line) & 1023) * W + (W - 1) - posInLine + ((line - 1) >> 31);
  
  double2 a = io[k];
  double2 b = conjugate(io[v]);
  double2 t = trig[g * 256 + me];
  
  X2(a, b);
  M(b, conjugate(t));
  X2(a, b);

  a = sq(a);
  b = sq(b);

  X2(a, b);
  M(b,  t);
  X2(a, b);
  
  io[k] = conjugate(a);
  io[v] = b;
}

KERNEL(256) csquare2K(global double2 *io, CONST double2 *trig)  { csquare(2048, io, trig); }
*/

void transposeCore(local double *lds, double2 *u) {
  uint me = get_local_id(0);
  for (int b = 0; b < 2; ++b) {
    if (b) { bar(); }
    for (int i = 0; i < 16; ++i) {
      uint l = i * 4 + me / 64;
      uint c = me % 64;
      lds[l * 64 + (c + l) % 64] = ((double *)(u + i))[b];
    }
    bar();
    for (int i = 0; i < 16; ++i) {
      uint c = i * 4 + me / 64;
      uint l = me % 64;
      ((double *)(u + i))[b] = lds[l * 64 + (c + l) % 64];
    }
  }
}

void transpose(uint W, uint H, local double *lds, CONST double2 *in, global double2 *out, CONST double2 *trig) {
  uint GW = W / 64, GH = H / 64;
  uint g = get_group_id(0), gx = g % GW, gy = g / GW;
  gy = (gy + gx) % GH;
  in   += gy * 64 * W + gx * 64;
  out  += gy * 64     + gx * 64 * H;
  uint me = get_local_id(0), mx = me % 64, my = me / 64;
  
  double2 u[16];
  for (int i = 0; i < 16; ++i) {
    uint p = (my + i * 4) * W + mx;
    u[i] = in[p];
  }

  transposeCore(lds, u);
  
  for (int i = 0; i < 16; ++i) {
    uint k = mul24(gy * 64 + mx, gx * 64 + my + (uint) i * 4);
    M(u[i], trig[(k & 127)]);
    M(u[i], trig[128 + ((k >> 7) & 127)]);
    M(u[i], trig[256 + (k >> 14)]);

    uint p = (my + i * 4) * H + mx;
    out[p] = u[i];
  }
}

// in place
/*
KERNEL(256) transp1K(global double2 *io, CONST double2 *trig) {
  uint W = 1024, GW = W / 64;
  uint g = get_group_id(0), gx = g % GW, gy = g / GW;
  io   += gy * 64 * W + gx * 64;
  uint me = get_local_id(0), mx = me % 64, my = me / 64;
  
  double2 u[16];
  for (int i = 0; i < 16; ++i) {
    uint p = (my + i * 4) * W + mx;
    u[i] = io[p];
  }

  local double lds[4096];
  transposeCore(lds, u);
  
  for (int i = 0; i < 16; ++i) {
    uint k = mul24(gy * 64 + mx, gx * 64 + my + (uint) i * 4);
    M(u[i], trig[(k & 127)]);
    M(u[i], trig[128 + ((k >> 7) & 127)]);
    M(u[i], trig[256 + (k >> 14)]);
        
    uint p = (my + i * 4) * W + mx;
    io[p] = u[i];
  }
}
*/

KERNEL(256) transpose1K(CONST double2 *in, global double2 *out, CONST double2 *trig) {
  local double lds[4096];
  transpose(1024, 2048, lds, in, out, trig);
}

KERNEL(256) transpose2K(CONST double2 *in, global double2 *out, CONST double2 *trig) {
  local double lds[4096];
  transpose(2048, 1024, lds, in, out, trig);
}

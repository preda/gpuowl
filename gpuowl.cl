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

void tabMul(SMALL_CONST double2 *trig, double2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  for (int i = 1; i < n; ++i) { M(u[i], trig[me / f + i * (256 / f)]); }
}

void shuffleMul(SMALL_CONST double2 *trig, local double *lds, double2 *u, uint n, uint f) {
  bar();
  shufl(lds,   u, n, f);
  tabMul(trig, u, n, f);
}

void shuffleMulBig(SMALL_CONST double2 *trig, local double2 *lds, double2 *u, uint n, uint f) {
  bar();
  shuflBig(lds, u, n, f);
  tabMul(trig,  u, n, f);
}

void fft1kBigLDS(local double2 *lds, double2 *u, SMALL_CONST double2 *trig1k) {
  fft4(u);
  shuflBig(lds,  u, 4, 64);
  tabMul(trig1k, u, 4, 64);
  
  fft4(u);
  shuffleMulBig(trig1k, lds, u, 4, 16);
  
  fft4(u);
  shuffleMulBig(trig1k, lds, u, 4, 4);

  fft4(u);
  shuffleMulBig(trig1k, lds, u, 4, 1);

  fft4(u);
}

void fft1kImpl(local double *lds, double2 *u, SMALL_CONST double2 *trig1k) {
  fft4(u);
  shufl(lds,     u, 4, 64);
  tabMul(trig1k, u, 4, 64);  
  
  fft4(u);
  shuffleMul(trig1k, lds, u, 4, 16);
  
  fft4(u);
  shuffleMul(trig1k, lds, u, 4, 4);

  fft4(u);
  shuffleMul(trig1k, lds, u, 4, 1);

  fft4(u);
}

void fft2kImpl(local double *lds, double2 *u, SMALL_CONST double2 *trig2k) {
  fft8(u);
  shufl(lds,   u, 8, 32);
  tabMul(trig2k, u, 8, 32);

  fft8(u);
  shuffleMul(trig2k, lds, u, 8, 4);

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
    M(u[i],     trig2k[i * 512 + me]);
    M(u[i + 4], trig2k[i * 512 + me + 256]);
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

long toLong(double x, float *maxErr) {
  return roundWithErr(x, maxErr);
  /*
  double rx = rint(x);
  *maxErr = max(*maxErr, fabs((float) (x - rx)));
  return rx;
  */
}

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

double updateD(double *carry, double a, int bits) {
  double x = a + *carry;
  *carry = rint(ldexp(x, -bits));
  return x - ldexp(*carry, bits);
}

int2 update(long *carry, long2 r, uchar2 bits) {
  int a = updateA(carry, r.x, bits.x);
  int b = updateB(carry, r.y, bits.y);
  return (int2) (a, b);
}

int2 car0(long *carry, double2 u, double2 a, uchar2 bits, float *maxErr) {
  long a0 = toLong(u.x * a.x, maxErr);
  long a1 = toLong(u.y * a.y, maxErr);
  return update(carry, (long2)(a0, a1), bits);
}

int2 car0Fast(long *carry, double2 u, double2 a, uchar2 bits) {
  long a0 = rint(u.x * a.x);
  long a1 = rint(u.y * a.y);
  return update(carry, (long2)(a0, a1), bits);
}

int2 car1(long *carry, int2 r, uchar2 bits) {
  return update(carry, (long2)(r.x, r.y), bits);
}

double2 car2(long carry, int2 r, uchar2 bits) {
  int a = updateA(&carry, r.x, bits.x);
  int b = r.y + (int) carry;
  return (double2) (a, b);
}

double2 dar0(double *carry, double2 u, double2 ia, float *maxErr, uint baseBits) {
  double a = updateD(carry, roundWithErr(u.x * fabs(ia.x), maxErr), baseBits + (ia.x < 0));
  double b = updateD(carry, roundWithErr(u.y * fabs(ia.y), maxErr), baseBits + (ia.y < 0));
  return (double2)(a, b);
}

double2 dar2(double carry, double2 r, double2 a, uint baseBits) {
  double x = updateD(&carry, r.x, baseBits + (a.x < 0));
  return (double2) (x, carry + r.y) * fabs(a);
}

KERNEL(256) mega1K(const uint baseBitlen, global double2 *io, volatile global double *transfer, volatile global uint *ready,
                   volatile global uint *globalErr,
                   CONST double2 *A, CONST double2 *iA, SMALL_CONST double2 *trig1k) {

  
  uint gr = get_group_id(0);
  uint gm = gr % 2048;
  uint me = get_local_id(0);
  uint step = gm * 1024;
  
  io     += step;
  A      += step;
  iA     += step;
  transfer += ((int)gr - 1) * 1024;

  double2 u[4];
  for (int i = 0; i < 4; ++i) { u[i] = io[i * 256 + me]; }

#ifdef LOW_LDS
  local double lds[1024];
  fft1kImpl(lds, u, trig1k);
#else
  local double2 lds[1024];
  fft1kBigLDS(lds, u, trig1k);
#endif
  
  float err = 0;
  #pragma unroll 1
  for (int i = 0; i < 4; ++i) {
    double carry = (i == 0 && gm == 0 && me == 0) ? -2 : 0;
    uint p = i * 256 + me;
    u[i] = dar0(&carry, conjugate(u[i]), iA[p], &err, baseBitlen);
    if (gr < 2048) { transfer[1024 + p] = carry; }
  }
  
  local uint *maxErr = (local uint *) lds;
  if (me == 0) { *maxErr = 0; }
  
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  if (gr < 2048 && me == 0) { atomic_xchg(&ready[gr], 1); }
  if (gr == 0) { return; }
  atomic_max(maxErr, *(uint *)&err);
  
  if (me == 0) { while(!atomic_xchg(&ready[gr - 1], 0)); }
  bar();

  if (me == 0) { atomic_max(globalErr, *maxErr); }
  
  for (int i = 0; i < 4; ++i) {
    uint p = i * 256 + me;
    double carry = transfer[(p - gr / 2048) & 1023];
    u[i] = dar2(carry, u[i], A[p], baseBitlen);
  }

#ifdef LOW_LDS
  fft1kImpl(lds, u, trig1k);
#else
  fft1kBigLDS(lds, u, trig1k);
#endif
  
  for (int i = 0; i < 4; ++i) { io[i * 256 + me]  = u[i]; }
}

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

// conjugates input
KERNEL(256) carryA(CONST double2 *in, CONST double2 *A, global int2 *out, global long *carryOut,
                 CONST uchar2 *bitlen, global uint *globalMaxErr) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);

  uint step = g % 4 * 256 + g / 4 * 8 * 1024;
  in     += step;
  A      += step;
  out    += step;
  bitlen += step;

  float maxErr = 0;
  long carry = (g == 0 && me == 0) ? -2 : 0;

  for (int i = 0; i < 8; ++i) {
    uint p = me + i * 1024;
    out[p] = car0(&carry, conjugate(in[p]), fabs(A[p]), bitlen[p], &maxErr);
  }

  carryOut[me + g * 256] = carry;

  local uint localMaxErr;
  if (me == 0) { localMaxErr = 0; }
  bar();
  atomic_max(&localMaxErr, *(uint *)&maxErr);
  bar();
  if (me == 0) { atomic_max(globalMaxErr, localMaxErr); }
}

void carryBCore(uint H, global int2 *in, CONST long *carryIn, CONST uchar2 *bitlen, global uint *maxErr) {
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
  
  if (carry) { atomic_max(maxErr, (3 << 28)); }  // Assert no carry left at this point.
}

KERNEL(256) carryB_2K(global int2 *in, global long *carryIn, CONST uchar2 *bitlen, global uint *maxErr) {
  carryBCore(2048, in, carryIn, bitlen, maxErr);
}

// Inputs normal (non-conjugate); outputs conjugate.
void csquare(uint W, global double2 *io, CONST double2 *trig) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);

  uint line = g / (W / 512);
  uint posInLine = g % (W / 512) * 256 + me;
  uint k = line * W + posInLine + ((line - 1) >> 31);
  uint v = ((1024 - line) & 1023) * W + (W - 1) - posInLine;
  
  double2 a = io[k];
  double2 b = conjugate(io[v]);
  double2 t = trig[g * 256 + me]; //equiv: [line * (W / 2) + posInLine];
  
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
  
  if (g == 0 && me == 0) {
    a = conjugate(io[0]);
    double t = a.x * a.y;
    a *= a;
    io[0] = (double2)((a.x + a.y) * 8, t * 16);
  }
}

KERNEL(256) csquare2K(global double2 *io, CONST double2 *trig)  { csquare(2048, io, trig); }

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
  trig += (gy + gx * GH) * (64 * 64);
  
  uint me = get_local_id(0), mx = me % 64, my = me / 64;
  
  double2 u[16];
  for (int i = 0; i < 16; ++i) {
    uint p = (my + i * 4) * W + mx;
    u[i] = in[p];
  }

  transposeCore(lds, u);
  
  for (int i = 0; i < 16; ++i) {
    uint p = (my + i * 4) * H + mx;
    out[p] = mul(u[i], trig[i * 256 + me]);
  }
}

// in place
KERNEL(256) transp1K(global double2 *io, CONST double2 *trig) {
  uint W = 1024, H = 2048, GW = W / 64;
  uint g = get_group_id(0), gx = g % GW, gy = g / GW;
  io   += gy * 64 * W + gx * 64;
  trig += g * (64 * 64);
  
  uint me = get_local_id(0), mx = me % 64, my = me / 64;
  
  double2 u[16];
  for (int i = 0; i < 16; ++i) {
    uint p = (my + i * 4) * W + mx;
    u[i] = mul(io[p], trig[i * 256 + me]);
  }

  local double lds[4096];
  transposeCore(lds, u);
  
  for (int i = 0; i < 16; ++i) {
    uint p = (my + i * 4) * W + mx;
    io[p] = u[i];
  }
}

/* // not used
KERNEL(256) transpose1K(CONST double2 *in, global double2 *out, CONST double2 *trig) {
  local double lds[4096];
  transpose(1024, 2048, lds, in, out, trig);
}
*/

KERNEL(256) transpose2K(CONST double2 *in, global double2 *out, CONST double2 *trig) {
  local double lds[4096];
  transpose(2048, 1024, lds, in, out, trig);
}

/*
KERNEL(256) mega1K(double2 *io, global long *gCarry, global uint *ready, global int *globalErr,
                   CONST double2 *A, CONST double2 *iA, CONST uchar2 *bitlen, SMALL_CONST double2 *trig1k) {
  local double lds[1024];
  
  uint gr = get_group_id(0);
  uint gm = gr % 1024;
  uint me = get_local_id(0);

  io     += gm * 2048;
  A      += gm * 2048;
  iA     += gm * 2048;
  bitlen += gm * 2048;

  double2 u[4];
  for (int i = 0; i < 4; ++i) { u[i] = io[i * 256 + me]; }

  fft1kImpl(lds, u, trig1k);

  float err = 0;
  int2 r[8];
  long carry[4];
  for (int i = 0; i < 4; ++i) {
    carry[i] = (i == 0 && g == 0 && me == 0) ? -2 : 0;
    uint p   = i * 256 + me;
    r[i]     = car0(carry + i, conjugate(u[i]), iA[p], bitlen[p], &err);
    u[i]     = io[p + 1024];
  }

  fft1kImpl(lds, u, trig1k);
  
  int2 *r = (int2 *) &u;
  for (int i = 0; i < 4; ++i) {
    uint p   = i * 256 + me + 1024;
    r[i + 4] = car0(carry + i, conjugate(u[i]), iA[p], bitlen[p], &err);
    gCarry[gm * 1024 + i * 256 + me] = carry[i];
  }

  local uint *maxErr = &lds;
  if (me == 0) { *maxErr = 0; }
  
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  if (me == 0) { ready[gr] = 1; }
  if (gr == 0) { return; }
  atomic_max(maxErr, (uint) (err * (1 << 30)));
  
  bar();
  if (me == 0) {
    atomic_max(globalErr, *maxErr);
    while(!ready[gr - 1]);  // busy wait
    ready[gr - 1] = 0;
  }  
  bar();

  for (int i = 0; i < 4; ++i) {
    uint p = i * 256 + me;
    carry[i] = gCarry[(gr * 1024 + p - 1) % (1024 * 1024)];
    u[i] = car1(carry + i, r[i], bitlen[p]) * A[p];
  }

  fft1kImpl(lds, u, trig1k);

  for (int i = 0; i < 4; ++i) {
    io[i * 256 + me]  = u[i];
    uint p = i * 256 + 1024 + me;
    u[i]   = car2(carry + i, r[i + 4], bitlen[p]) * A[p];
  }

  fft1kImpl(lds, u, trig1K);

  for (int i = 0; i < 4; ++i) { io[i * 256 + 1024 + me] = u[i]; }
}
*/

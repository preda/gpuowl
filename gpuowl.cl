// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#define K(x, y) kernel __attribute__((reqd_work_group_size(x, y, 1))) void
#define CONST const global
// constant
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

void tabMul(SMALL_CONST double2 *trig, double2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  for (int i = 1; i < n; ++i) { M(u[i], trig[me / f + i * (256 / f)]); }
}

void shuffleMul(SMALL_CONST double2 *trig, local double *lds, double2 *u, uint n, uint f) {
  bar();
  shufl(lds, u, n, f);
  tabMul(trig, u, n, f);
}

void fft1kImpl(local double *lds, double2 *u, SMALL_CONST double2 *trig1k) {
  fft4(u);
  shufl(lds,   u, 4, 64);
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

K(256, 1) fftPremul1K(CONST int2 *in, global double2 *out, CONST double2 *A, SMALL_CONST double2 *trig1k) {
  uint g = get_group_id(0);
  uint step = g * 1024;
  in  += step;
  A   += step;
  out += step;
  
  uint me = get_local_id(0);
  double2 u[4];

  for (int i = 0; i < 4; ++i) {
    int2 r = in[me + i * 256];
    u[i] = (double2)(r.x, r.y) * A[me + i * 256];
  }

  local double lds[1024];
  fft1kImpl(lds, u, trig1k);

  for (int i = 0; i < 4; ++i) { out[me + i * 256] = u[i]; }  
}

K(256, 1) fft1K(global double2 *in, SMALL_CONST double2 *trig1k) {
  uint g = get_group_id(0);
  in += g * 1024;
  
  uint me = get_local_id(0);
  double2 u[4];

  for (int i = 0; i < 4; ++i) { u[i] = in[me + i * 256]; }

  local double lds[1024];
  fft1kImpl(lds, u, trig1k);

  for (int i = 0; i < 4; ++i) { in[me + i * 256] = u[i]; }  
}

// Input is transposed.
void fft1Kt(uint W, CONST double2 *in, global double2 *out, SMALL_CONST double2 *trig1k) {
  uint g = get_group_id(0);
  in  += g % (W / 64) * 64 + g / (W / 64) * W;

  uint me = get_local_id(0);
  double2 u[4];

  for (int i = 0; i < 4; ++i) { u[i] = in[me % 64 + (me / 64 + i * 4) * 64 * W]; }

  local double lds[1024];
  fft1kImpl(lds, u, trig1k);
  
  uint lg = g / (W / 64) + g % (W / 64) * 64;
  out += lg * 1024;

  for (int i = 0; i < 4; ++i) { out[i * 256 + me] = u[i]; }
}

// Outputs conjugate (used in the inverse FFT). Input is transposed.
void cfft1Kt(uint W, CONST double2 *in, global double2 *out, SMALL_CONST double2 *trig1k) {
  uint g = get_group_id(0);
  in  += g % (W / 64) * 64 + g / (W / 64) * W;

  uint me = get_local_id(0);
  double2 u[4];

  for (int i = 0; i < 4; ++i) { u[i] = in[me % 64 + (me / 64 + i * 4) * 64 * W]; }

  local double lds[1024];
  fft1kImpl(lds, u, trig1k);
  
  uint lg = g / (W / 64) + g % (W / 64) * 64;
  out += lg * 1024;

  for (int i = 0; i < 4; ++i) { out[i * 256 + me] = conjugate(u[i]); }
}

K(256, 1) fft1K_1K(CONST double2 *in, global double2 *out, SMALL_CONST double2 *trig1k) { fft1Kt(1024, in, out, trig1k); }
K(256, 1) cfft1K_1K(CONST double2 *in, global double2 *out, SMALL_CONST double2 *trig1k) { cfft1Kt(1024, in, out, trig1k); }
K(256, 1) cfft1K_2K(CONST double2 *in, global double2 *out, SMALL_CONST double2 *trig1k) { cfft1Kt(2048, in, out, trig1k); }

K(256, 1) fft2K(global double2 *in, SMALL_CONST double2 *trig2k) {
  uint g = get_group_id(0);
  in += g * 2048;
  
  uint me = get_local_id(0);
  double2 u[8];

  for (int i = 0; i < 8; ++i) { u[i] = in[me + i * 256]; }

  local double lds[2048];
  fft2kImpl(lds, u, trig2k);

  for (int i = 0; i < 4; ++i) {
    in[me + i * 512]       = u[i];
    in[me + i * 512 + 256] = u[i + 4];
  }  
}

// input 1024/64, output 2048.
K(256, 1) fft2K_1K(CONST double2 *in, global double2 *out, SMALL_CONST double2 *trig2k) {
  uint g = get_group_id(0);
  in  += g % 16 * 64 + g / 16 * 1024;

  uint me = get_local_id(0);
  double2 u[8];

  for (int i = 0; i < 8; ++i) { u[i] = in[me % 64 + (me / 64 + i * 4) * 64 * 1024]; }

  local double lds[2048];
  fft2kImpl(lds, u, trig2k);
  
  uint lg = g / 16 + g % 16 * 64;
  out += lg * 2048;

  for (int i = 0; i < 4; ++i) {
    out[i * 512 + me]       = u[i];
    out[i * 512 + me + 256] = u[i + 4];
  }
}

long toLong(double x, float *maxErr) {
  double rx = rint(x);
  *maxErr = max(*maxErr, fabs((float) (x - rx)));
  return rx;
}

int lowBits(long u, uint bits) { return (((int) ((uint) ((ulong) u))) << (32 - bits)) >> (32 - bits); }

int update(long *carry, long x, uint bits) {
  long u = *carry + x;
  int w = lowBits(u, bits);
  *carry = (u - w) >> bits;
  return w;
}

int2 car0(long *carry, double2 u, double2 a, uchar2 bits, float *maxErr) {
  int r0 = update(carry, toLong(u.x * a.x, maxErr), bits.x);
  int r1 = update(carry, toLong(u.y * a.y, maxErr), bits.y);
  return (int2) (r0, r1);
}

int2 car1(long *carry, int2 r, uchar2 bits) {
  int a = update(carry, r.x, bits.x);
  int b = update(carry, r.y, bits.y);
  return (int2) (a, b);
}

K(256, 1) carryA(CONST double2 *in, CONST double2 *A, global int2 *out, global long *carryOut,
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
    out[p] = car0(&carry, in[p], A[p], bitlen[p], &maxErr);
  }

  carryOut[me + g * 256] = carry;

  local uint localMaxErr;
  if (me == 0) { localMaxErr = 0; }
  bar();
  atomic_max(&localMaxErr, maxErr * (1 << 30));
  bar();
  if (me == 0) { atomic_max(globalMaxErr, localMaxErr); }
}

void carryBCore(uint H, global int2 *in, global long *carryIn, CONST uchar2 *bitlen) {
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
    if (!carry) { break; }
  }
  // if (carry) { globalmaxErr = 0.5; } // Assert no carry at this point.
}

K(256, 1) carryB_1K(global int2 *in, global long *carryIn, CONST uchar2 *bitlen) { carryBCore(1024, in, carryIn, bitlen); }
K(256, 1) carryB_2K(global int2 *in, global long *carryIn, CONST uchar2 *bitlen) { carryBCore(2048, in, carryIn, bitlen); }

// Inputs normal (non-conjugate); outputs conjugate.
void csquare(uint W, global double2 *in, CONST double2 *trig) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);

  uint line = g / (W / 512);
  uint posInLine = g % (W / 512) * 256 + me;
  uint k = line * W + posInLine + ((line - 1) >> 31);
  uint v = ((1024 - line) & 1023) * W + (W - 1) - posInLine;
  
  double2 a = in[k];
  double2 b = conjugate(in[v]);
  double2 t = trig[g * 256 + me]; //equiv: [line * (W / 2) + posInLine];
  
  X2(a, b);
  M(b, conjugate(t));
  X2(a, b);

  a = sq(a);
  b = sq(b);

  X2(a, b);
  M(b,  t);
  X2(a, b);
  
  in[k] = conjugate(a);
  in[v] = b;
  
  if (g == 0 && me == 0) {
    a = conjugate(in[0]);
    double t = a.x * a.y;
    a *= a;
    in[0] = (double2)((a.x + a.y) * 8, t * 16);
  }
}

K(256, 1) csquare2K(global double2 *in, CONST double2 *trig)  { csquare(2048, in, trig); }
K(256, 1) csquare1K(global double2 *in, CONST double2 *trig)  { csquare(1024, in, trig); }

void transposeCore(double2 *u) {
  local double lds[4096];
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

K(256, 1) transpose1K(global double2 *in, CONST double2 *trig) {
  uint W = 1024;
  uint g = get_group_id(0);
  uint gx = g % (W / 64);
  uint gy = g / (W / 64);
  in   += gy * 64 * W + gx * 64;
  trig += g * 4096;
  
  uint me = get_local_id(0);
  uint mx = me % 64;
  uint my = me / 64;
  
  double2 u[16];
  for (int i = 0; i < 16; ++i) {
    uint p = (my + i * 4) * W + mx;
    u[i] = mul(in[p], trig[i * 256 + me]);
  }

  transposeCore(u);
  
  for (int i = 0; i < 16; ++i) {
    uint p = (my + i * 4) * W + mx;
    in[p] = u[i];
  }
}

// transpose with multiplication on output.
K(256, 1) mtranspose2K(global double2 *in, CONST double2 *trig) {
  uint W = 2048;
  uint g = get_group_id(0);
  uint gx = g % (W / 64);
  uint gy = g / (W / 64);
  in   += gy * 64 * W + gx * 64;
  trig += (gy + gx * 16) * 4096;
  
  uint me = get_local_id(0);
  uint mx = me % 64;
  uint my = me / 64;
  
  double2 u[16];
  for (int i = 0; i < 16; ++i) {
    uint p = (my + i * 4) * W + mx;
    u[i] = in[p];
  }

  transposeCore(u);
  
  for (int i = 0; i < 16; ++i) {
    uint p = (my + i * 4) * W + mx;
    in[p] = mul(u[i], trig[i * 256 + me]);
  }
}


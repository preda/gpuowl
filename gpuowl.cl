// gpuOwL, an OpenCL Mersenne primality checker.
// Copyright (C) 2017 Mihai Preda.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

// "overloadable" would be really useful, but it works on LLVM only.
#define OVERLOAD __attribute__((overloadable))

typedef global double2 * restrict d2ptr;
typedef global int2 * restrict i2ptr;

// add-sub: (a, b) = (a + b, a - b)
#define X1(a, b) { double  t = a; a = t + b; b = t - b; }
#define X2(a, b) { double2 t = a; a = t + b; b = t - b; }

// swap: (a, b) = (b, a)
#define S2(a, b) { double2 t = a; a = b; b = t; }
double2 swap(double2 u) { return (double2) (u.y, u.x); }

void bar()    { barrier(CLK_LOCAL_MEM_FENCE); }
void bigBar() { barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); }

// complex multiplication.
double2 muld(double2 u, double a, double b) { return (double2) { u.x * a - u.y * b, u.x * b + u.y * a}; }
double2 mul(double2 u, double2 v) { return muld(u, v.x, v.y); }

// mutating complex multiplication.
#define MUL(x, a, b) x = muld(x, a, b)
#define M(x, t) x = mul(x, t)

#define MUL_2(x, a, b) x = muld(x, a, b) * M_SQRT1_2

// complex square.
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

void tabMul(const double2 *trig, double2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  for (int i = 1; i < n; ++i) { M(u[i], trig[me / f + i * (256 / f)]); }
}

void fft1kImpl(local double *lds, double2 *u, const double2 *trig) {
  fft4(u);
  shufl(lds,      u, 4, 64);
  tabMul(trig, u, 4, 64);
  
  fft4(u);
  bar();
  shufl(lds,      u, 4, 16);
  tabMul(trig, u, 4, 16);
  
  fft4(u);
  bar();
  shufl(lds,      u, 4, 4);
  tabMul(trig, u, 4, 4);

  fft4(u);
  bar();
  shufl(lds,      u, 4, 1);
  tabMul(trig, u, 4, 1);

  fft4(u);
}

void fft2kImpl(local double *lds, double2 *u, const double2 *trig) {
  fft8(u);
  shufl(lds,      u, 8, 32);
  tabMul(trig, u, 8, 32);

  fft8(u);
  bar();
  shufl(lds,      u, 8, 4);
  tabMul(trig, u, 8, 4);
  
  fft8(u);

  uint me = get_local_id(0);
  for (int b = 0; b < 2; ++b) {
    bar();
    for (int i = 0; i < 8; ++i) { lds[(me + i * 256) / 4 + me % 4 * 512] = ((double *) (u + i))[b]; }
    bar();
    for (int i = 0; i < 4; ++i) {
      ((double *) (u + i))[b]     = lds[i * 512       + me];
      ((double *) (u + i + 4))[b] = lds[i * 512 + 256 + me];
    }
  }

  for (int i = 1; i < 4; ++i) {
    M(u[i],     trig[i * 512       + me]);
    M(u[i + 4], trig[i * 512 + 256 + me]);
  }
     
  fft4(u);
  fft4(u + 4);

  // fix order: interleave u[0:4] and u[4:8], like (u.even, u.odd) = (u.lo, u.hi).
  S2(u[1], u[2]);
  S2(u[1], u[4]);
  S2(u[5], u[6]);
  S2(u[3], u[6]);
}

void fftImpl(uint N, local double *lds, double2 *u, const double2 *trig) {
  if (N == 4) { fft1kImpl(lds, u, trig); } else { fft2kImpl(lds, u, trig); }
}

void read(uint N, double2 *u, global double2 *in, uint base) {
  for (int i = 0; i < N; ++i) { u[i] = in[base + i * 256 + (uint) get_local_id(0)]; }
}

void write(uint N, double2 *u, global double2 *out, uint base) {
  for (int i = 0; i < N; ++i) { out[base + i * 256 + (uint) get_local_id(0)] = u[i]; }
}

// FFT of size N * 256.
void fft(uint N, local double *lds, double2 *u, global double2 *io, const double2 *trig) {
  uint g = get_group_id(0);
  uint step = g * (N * 256);
  io += step;

  read(N, u, io, 0);
  
  fftImpl(N, lds, u, trig);

  write(N, u, io, 0);
}

double2 toDouble(int2 r) { return (double2) (r.x, r.y); }

// fftPremul: weight words with "A" (for IBDWT) followed by FFT.
void fftPremul(uint N, local double *lds, double2 *u, const int2 *in, global double2 *out, const double2 *A, const double2 *trig) {
  uint g = get_group_id(0);
  uint step = g * (N * 256);
  in  += step;
  A   += step;
  out += step;
  
  uint me = get_local_id(0);

  for (int i = 0; i < N; ++i) { u[i] = toDouble(in[me + i * 256]) * fabs(A[me + i * 256]); }

  fftImpl(N, lds, u, trig);

  write(N, u, out, 0);
}

KERNEL(256) fft1K(d2ptr io, const d2ptr trig1k) {
  local double lds[4 * 256];
  double2 u[4];
  fft(4, lds, u, io, trig1k);
}

KERNEL(256) fft2K(d2ptr io, const d2ptr trig2k) {
  local double lds[8 * 256];
  double2 u[8];
  fft(8, lds, u, io, trig2k);
}

KERNEL(256) fftPremul1K(const i2ptr in, d2ptr out, const d2ptr A, const d2ptr trig1k) {
  local double lds[4 * 256];
  double2 u[4];
  fftPremul(4, lds, u, in, out, A, trig1k);
}

KERNEL(256) fftPremul2K(const i2ptr in, d2ptr out, const d2ptr A, const d2ptr trig2k) {
  local double lds[8 * 256];
  double2 u[8];
  fftPremul(8, lds, u, in, out, A, trig2k);
}

// Round x to long.
long toLong(double x) { return rint(x); }

int lowBits(int u, uint bits) { return (u << (32 - bits)) >> (32 - bits); }

// Carry propagation, with optional MUL.
int updateMul(int mul, long *carry, long x, uint bits) {
  long u = *carry + x * mul;
  int w = lowBits(u, bits);
  *carry = (u - w) >> bits;
  return w;
}

// Simpler version of signbit(a).
uint signBit(double a) { return ((uint *)&a)[1] >> 31; }

// double xorSign(double a, uint bit) { ((uint *)&a)[1] ^= (bit << 31); return a; }
// double2 xorSign2(double2 a, uint bit) { return (double2) (xorSign(a.x, bit), xorSign(a.y, bit)); }

double2 xorSign2(double2 a, uint bit) { return bit ? -a : a; }

uint bitlen(uint base, double a) { return base + signBit(a); }

// Reverse weighting, round, carry propagation for a pair of doubles; with optional MUL.
int2 car0Mul(int mul, long *carry, double2 u, double2 ia, uint baseBits) {  
  u *= fabs(ia); // Reverse weighting by multiply with "ia"
  int a = updateMul(mul, carry, toLong(u.x), bitlen(baseBits, ia.x));
  int b = updateMul(mul, carry, toLong(u.y), bitlen(baseBits, ia.y));
  return (int2) (a, b);
}

// Carry propagation.
int2 car1(long *carry, int2 r, double2 ia, uint base) {
  int a = updateMul(1, carry, r.x, bitlen(base, ia.x));
  int b = updateMul(1, carry, r.y, bitlen(base, ia.y));
  return (int2) (a, b);
}

// Returns a pair (reduced x, carryOut).
double2 carryStep(double x, int bits) {
  double carry = rint(ldexp(x, -bits));
  return (double2)(x - ldexp(carry, bits), carry);
}

// Applies inverse weight "iA" and rounding, and propagates carry over the two words.
double2 weightAndCarry(double *carry, double2 u, double2 iA, uint baseBits) {
  u = rint(u * fabs(iA)); // reverse weight and round.
  double2 r0 = carryStep(*carry + u.x, bitlen(baseBits, iA.x));
  double2 r1 = carryStep(r0.y   + u.y, bitlen(baseBits, iA.y));
  *carry = r1.y;
  return (double2) (r0.x, r1.x);
}

double2 carryAndWeight(double *carry, double2 u, double2 a, uint baseBits) {
  double2 r0 = carryStep(*carry + u.x, bitlen(baseBits, a.x));
  double2 r1 = carryStep(r0.y   + u.y, bitlen(baseBits, a.y));
  *carry = r1.y;
  return (double2) (r0.x, r1.x) * fabs(a);
}

// No carry out. The final carry is "absorbed" in the last word.
double2 carryAndWeightFinal(double carry, double2 u, double2 a, uint baseBits) {
  double2 r = carryStep(carry + u.x, bitlen(baseBits, a.x));
  return (double2) (r.x, r.y + u.y) * fabs(a);
}

// The "carryConvolution" is equivalent to the sequence: fft, carryA, carryB, fftPremul.
// It uses "stareway" carry data forwarding from group K to group K+1.
// N gives the FFT size, W = N * 256.
// H gives the nuber of "lines" of FFT.
void carryConvolution(uint N, uint H, local double *lds, double2 *u,
                  uint baseBitlen,
                  global double2 *io, global double *carryShuttle, volatile global uint *ready,
                  const double2 *A, const double2 *iA, const double2 *trig) {
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
    double carry = 0;
    u[i] = weightAndCarry(&carry, conjugate(u[i]), iA[p], baseBitlen);
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
    double carry = carryShuttle[(gr - 1) * W + ((p - gr / H) & (W - 1))];
    u[i] = carryAndWeightFinal(carry, u[i], A[p], baseBitlen);
  }

  fftImpl(N, lds, u, trig);
  write(N, u, io, 0);
}

KERNEL(256) carryConv1K_2K(uint baseBitlen, d2ptr io,
                           global double * restrict carryShuttle, volatile global uint * restrict ready,
                           const d2ptr A, const d2ptr iA, const d2ptr trig1k) {
  local double lds[4 * 256];
  double2 u[4];
  carryConvolution(4, 2048, lds, u, baseBitlen, io, carryShuttle, ready, A, iA, trig1k);
}

#ifdef ENABLE_BUG

KERNEL(256) carryConv2K_2K(uint baseBitlen, d2ptr io,
                           global double * restrict carryShuttle, volatile global uint * restrict ready,
                           const d2ptr A, const d2ptr iA, const d2ptr trig1k) {
  local double lds[8 * 256];
  double2 u[8];
  carryConvolution(8, 2048, lds, u, baseBitlen, io, carryShuttle, ready, A, iA, trig1k);
}

#endif

double2 addsub(double2 a) { return (double2) (a.x + a.y, a.x - a.y); }

double2 foo2(double2 a, double2 b) {
  a = addsub(a);
  b = addsub(b);
  return addsub(a * b);
}

// computes 2*[x^2+y^2 + i*(2*x*y)]. Needs a name.
double2 foo(double2 a) { return foo2(a, a); }

void reverse(local double2 *lds, double2 *u, bool bump) {
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

// This kernel is equivalent to the sequence: fft2K, csquare2K, fft2K.
// It does less global memory transfers, but uses more VGPRs.
KERNEL(256) tail(d2ptr io, const d2ptr trig, const d2ptr bigTrig) {
  uint g = get_group_id(0);
  uint me = get_local_id(0);
  local double lds[2048];
  
  double2 u[8];
  read(8, u, io, g * 2048);
  fftImpl(8, lds, u, trig);

  reverse((local double2 *) lds, u, g == 0);

  double2 v[8];
  uint line2 = g ? 1024 - g : 512;
  read(8, v, io, line2 * 2048);
  bar(); fftImpl(8, lds, v, trig);

  reverse((local double2 *) lds, v, false);
  
  if (g == 0) { for (int i = 0; i < 4; ++i) { S2(u[4 + i], v[4 + i]); } }

  double2 tt = bigTrig[4096 + 512 + g];
  for (int i = 0; i < 4; ++i) {
    double2 a = u[i];
    double2 b = conjugate(v[4 + i]);
    double2 t = swap(mul(tt, bigTrig[256 * i + me]));
    if (i == 0 && g == 0 && me == 0) {
      a = 4 * foo(a);
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

  tt = bigTrig[4096 + 512 +  line2];
  for (int i = 0; i < 4; ++i) {
    double2 a = v[i];
    double2 b = conjugate(u[4 + i]);
    double2 t = swap(mul(tt, bigTrig[256 * i + me]));
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

  reverse((local double2 *) lds, u, g == 0);
  bar(); fftImpl(8, lds, u, trig);

  write(8, u, io, g * 2048);
  
  reverse((local double2 *) lds, v, false);
  bar(); fftImpl(8, lds, v, trig);

  write(8, v, io, line2 * 2048);
}

// Carry propagation with optional MUL, over 16 words.
// Input is doubles. They are weighted with the "inverse weight" A
// and rounded to output ints and to left-over carryOut.
void carryMul(const int mul, const uint baseBits,
              const double2 *in, const double2 *A, global int2 *out,
              global long *carryOut) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);

  uint step = g % 4 * 256 + g / 4 * 8 * 1024;
  in     += step;
  A      += step;
  out    += step;

  long carry = 0;

  for (int i = 0; i < 8; ++i) {    
    uint p = me + i * 1024;
    out[p] = car0Mul(mul, &carry, conjugate(in[p]), A[p], baseBits);
  }

  carryOut[me + g * 256] = carry;
}

// Carry propagation. conjugates input.
KERNEL(256) carryA(const uint baseBits, const d2ptr in, const d2ptr A, i2ptr out, global long *carryOut) {
  carryMul(1, baseBits, in, A, out, carryOut);
}

// Carry propagation + MUL 3. conjugates input.
KERNEL(256) carryMul3(const uint baseBits, const d2ptr in, const d2ptr A, i2ptr out, global long *carryOut) {
  carryMul(3, baseBits, in, A, out, carryOut);
}

// The second round of carry propagation (16 words), needed to "link the chain" after carryA.
// Input is int words and the left-over carry from carryA.
// Output is int words.
// The weights "A" are needed only to derive the bit size of each word (encoded in the sign of its elements).
void carryBCore(uint H, const uint baseBits, global int2 *io, const long *carryIn, const double2 *A) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);
  
  uint step = g % 4 * 256 + g / 4 * 8 * 1024;
  io += step;
  A  += step;
  
  uint prev = (g / 4 + (g % 4 * 256 + me) * (H / 8) - 1) & ((H / 8) * 1024 - 1);
  uint line = prev % (H / 8);
  uint col  = prev / (H / 8);
  long carry = carryIn[line * 1024 + col];
  
  for (int i = 0; i < 8; ++i) {
    uint p = me + i * 1024;
    io[p] = car1(&carry, io[p], A[p], baseBits);
    if (!carry) { return; }
  }
}

KERNEL(256) carryB_2K(const uint baseBits, i2ptr io, const global long * restrict carryIn, const d2ptr A) {
  carryBCore(2048, baseBits, io, carryIn, A);
}

// Inputs normal (non-conjugate); outputs conjugate.
void csquare(uint W, global double2 *io, const double2 *trig) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);

  if (g == 0 && me == 0) {
    io[0]    = 4 * foo(conjugate(io[0]));
    io[1024] = 8 * sq(conjugate(io[1024]));
    return;
  }
  
  uint line = g / (W / 512);
  uint posInLine = g % (W / 512) * 256 + me;
  uint k = line * W + posInLine;
  uint v = ((1024 - line) & 1023) * W + (W - 1) - posInLine + ((line - 1) >> 31);
  
  double2 a = io[k];
  double2 b = conjugate(io[v]);
  double2 t = swap(mul(trig[4096 + 512 + line], trig[posInLine]));
  
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

// Like csquare(), but for multiplication.
void cmul(uint W, global double2 *io, const double2 *in, const double2 *trig) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);

  if (g == 0 && me == 0) {
    io[0]    = 4 * (foo2(conjugate(io[0]), conjugate(in[0])));
    io[1024] = 8 * conjugate(mul(io[1024], in[1024]));
    return;
  }
  
  uint line = g / (W / 512);
  uint posInLine = g % (W / 512) * 256 + me;
  uint k = line * W + posInLine;
  uint v = ((1024 - line) & 1023) * W + (W - 1) - posInLine + ((line - 1) >> 31);
  
  double2 a = io[k];
  double2 b = conjugate(io[v]);
  double2 t = swap(mul(trig[4096 + 512 + line], trig[posInLine]));
  
  X2(a, b);
  M(b, conjugate(t));
  X2(a, b);
  
  double2 c = in[k];
  double2 d = conjugate(in[v]);
  X2(c, d);
  M(d, conjugate(t));
  X2(c, d);

  M(a, c);
  M(b, d);

  X2(a, b);
  M(b,  t);
  X2(a, b);
  
  io[k] = conjugate(a);
  io[v] = b;
}

KERNEL(256) csquare2K(d2ptr io, const d2ptr trig)  { csquare(2048, io, trig); }
KERNEL(256) cmul2K(d2ptr io, const d2ptr in, const d2ptr trig)  { cmul(2048, io, in, trig); }

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

void transpose(uint W, uint H, local double *lds, const double2 *in, global double2 *out, const double2 * trig) {
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
    M(u[i], trig[4096 + k % (W * H / 4096)]);
    M(u[i], trig[k / (W * H / 4096)]);

    uint p = (my + i * 4) * H + mx;
    out[p] = u[i];
  }
}

KERNEL(256) transpose1K_2K(const d2ptr in, d2ptr out, const d2ptr trig) {
  local double lds[4096];
  transpose(1024, 2048, lds, in, out, trig);
}

KERNEL(256) transpose2K_1K(const d2ptr in, d2ptr out, const d2ptr trig) {
  local double lds[4096];
  transpose(2048, 1024, lds, in, out, trig);
}

KERNEL(256) transpose2K_2K(const d2ptr in, d2ptr out, const d2ptr trig) {
  local double lds[4096];
  transpose(2048, 2048, lds, in, out, trig);
}

// in place
/*
KERNEL(256) transp1K(global double2 *io, const d2ptr trig) {
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

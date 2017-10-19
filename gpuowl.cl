// gpuOwl, an OpenCL Mersenne primality test.
// Copyright (C) 2017 Mihai Preda.

// Expected defines: WIDTH, HEIGHT, EXP.

#define NWORDS (WIDTH * HEIGHT * 2u)
#define STEP (NWORDS - (EXP & (NWORDS - 1)))
#define BASE_BITLEN (EXP / NWORDS)


// OpenCL 2.x introduces the "generic" memory space, so there's no need to specify "global" on pointers everywhere.
#if __OPENCL_C_VERSION__ >= 200
#define G
#else
#define G global
#endif


void bar()    { barrier(CLK_LOCAL_MEM_FENCE); }
void bigBar() { barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); }


#ifdef FFT_M31

// An GF(M31^2) NTT FFT convolution implementation.
// GF(p^2) is Gaussian integers ("complex integers") with the real/imaginary part modulo p.

typedef uint T;
typedef uint2 T2;

// Prime M(31) == 2**31 - 1
#define M31 0x7fffffffu

ulong u64(uint a) { return a; } // cast to 64 bits.


// input 32 bits except 2^32-1. Output 31 bits.
uint mod1(uint x) { return (x >> 31) + (x & M31); }
uint mod2(uint x, uint y) { return U2(mod1(x), mod1(y)); }

uint lo(ulong a) { return a & 0xffffffff; }
uint up(ulong a) { return a >> 32; }

// input 63 bits, output 31 bits.
uint bigmod1(ulong x) {
  x = u64(up(x) << 1) + lo(x); // 'up' is 31 bits.
  return mod1(lo(x) + (up(x) << 1));
}
uint bigmod2(ulong x, ulong y) { return U2(bigmod1(x), bigmod1(y)); }

uint add1(uint a, uint b) { return mod1(a + b); }
uint sub1(uint a, uint b) { return mod1(a + (M31 - b)); }

uint2 add(uint2 a, uint2 b) { return U2(add1(a.x, b.x), add1(a.y, b.y)); }
uint2 sub(uint2 a, uint2 b) { return U2(sub1(a.x, b.x), sub1(a.y, b.y)); }

uint shl(uint a, uint k) { return bigmod1(u64(a) << k); }
// uint2 shl2(uint2 a, uint2 k) { return U2(shl(a.x, k.x), shl(a.y, k.y)); }

ulong rawMul(uint a, uint b) { return u64(a) * b; }
uint mul(uint a, uint b) { return mod63(rawMul(a, b)); }

// Complex mul. input and output 31 bits.
uint2 mul(uint2 u, uint2 v) {
  uint a = u.x, b = u.y, c = v.x, d = v.y;
  ulong k1 = rawMul(c, add(a, b));
  ulong k2 = rawMul(a, sub(d, c));
  ulong k3 = rawMul(b, M31 - add(d, c));
  // k1..k3 have at most 62 bits, so sums are at most 63 bits.
  return U2(bigmod(k1 + k3), bigmod(k1 + k2));
}

// mul with (0, 1). (twiddle of tau/4, sqrt(-1) aka "i").
uint2 mul_t4(uint2 a) { return U2(M31 - a.y, a.x); }

// mul with (2^15, 2^15). (twiddle of tau/8 aka sqrt(i)).
uint2 mul_t8(uint2 a) { return U2(shl(a.x + (M31 - a.y), 15), shl(a.x + a.y, 15)); }

// mul with (-2^15, 2^15). (twiddle of 3*tau/8).
uint2 mul_3t8(uint2 ) { return U2(shl((M31 - a.x) + (M31 - a.y), 15), shl(a.x + (M31 - a.y), 15)); }

// input, output 31 bits. Uses (a + i*b)^2 == ((a+b)*(a-b) + i*2*a*b).
uint2 sq(uint2 a) { return U2(mul(a + b, sub(a, b)), mul(a, b << 1)); }


#endif


#if defined(FFT_DP)

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT ON

typedef double T;
typedef double2 T2;

#elif defined(FFT_SP)

typedef double T;
typedef double2 T2;

#endif


#if defined(FFT_DP) || defined(FFT_SP)

T add1(T a, T b) { return a + b; }
T sub1(T a, T b) { return a - b; }

T2 add(T2 a, T2 b) { return a + b; }
T2 sub(T2 a, T2 b) { return a - b; }

#endif


T2 U2(T a, T b) { return (T2)(a, b); }
T2 addsub(T2 a) { return U2(add1(a.x, a.y), sub1(a.x, a.y)); }


#if defined(FFT_DP) || defined(FFT_SP)

// complex mul
T2 mul(T2 a, T2 b) { return U2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

// complex square
T2 sq(T2 a) {
  T t = a.x * a.y * 2;
  a = addsub(a);
  return U2(a.x * a.y, t);
}

T2 mul_t4(T2 a)  { return mul(a, U2(0,  -1)); }
T2 mul_t8(T2 a)  { return mul(a, U2(1,  -1)) * (T)(M_SQRT1_2); }
T2 mul_3t8(T2 a) { return mul(a, U2(-1, -1)) * (T)(M_SQRT1_2); }

#endif


#define X2(a, b) { T2 t = a; a = add(t, b); b = sub(t, b); }
#define S2(a, b) { T2 t = a; a = b; b = t; }

T2 swap(T2 a) { return U2(a.y, a.x); }

T2 conjugate(T2 a) { return U2(a.x, -a.y); }

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
  S2(u[1], u[2]);
}

void fft8(T2 *u) {
  fft8Core(u);
  S2(u[1], u[4]);
  S2(u[3], u[6]);
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

  for (int i = 1; i < 4; ++i) {
    u[i]     = mul(u[i],     trig[i * 512       + me]);
    u[i + 4] = mul(u[i + 4], trig[i * 512 + 256 + me]);
  }
     
  fft4(u);
  fft4(u + 4);

  // fix order: interleave u[0:4] and u[4:8], like (u.even, u.odd) = (u.lo, u.hi).
  S2(u[1], u[2]);
  S2(u[1], u[4]);
  S2(u[5], u[6]);
  S2(u[3], u[6]);
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

double2 toDouble(int2 r) { return (double2) (r.x, r.y); }

// fftPremul: weight words with "A" (for IBDWT) followed by FFT.
void fftPremul(uint N, local double *lds, double2 *u, const G int2 *in, G double2 *out, const G double2 *A,
               const G double2 *trig) {
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

void reverse8(local double2 *lds, double2 *u, bool bump) {
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

void reverse4(local double2 *lds, double2 *u, bool bump) {
  uint me = get_local_id(0);
  uint rm = 255 - me + bump;
  
  bar();

  lds[rm + 0 * 256] = u[3];
  lds[bump ? ((rm + 256) & 511) : (rm + 256)] = u[2];
  
  bar();
  u[2] = lds[me];
  u[3] = lds[me + 256];
}

void reverse(uint N, local double2 *lds, double2 *u, bool bump) {
  if (N == 4) {
    reverse4(lds, u, bump);
  } else {
    reverse8(lds, u, bump);
  }
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

uint bitlen(uint base, double a) { return EXP / NWORDS + signBit(a); }

uint extra(uint k) { return mul24(k, STEP) & (NWORDS - 1); }

// Is the word at pos a big word (BASE_BITLEN+1 bits)? (vs. a small, BASE_BITLEN bits word).
bool isBigWord(uint k) { return extra(k) + STEP < NWORDS; }

// Number of bits for the word at pos.
uint bitlenAtPos(uint k) { return EXP / NWORDS + isBigWord(k); }

// Reverse weighting, round, carry propagation for a pair of doubles; with optional MUL.
int2 car0Mul(int mul, long *carry, double2 u, double2 ia, uint baseBits) {  
  u *= fabs(ia); // Reverse weighting by multiply with "ia"
  int a = updateMul(mul, carry, toLong(u.x), bitlen(baseBits, ia.x));
  int b = updateMul(mul, carry, toLong(u.y), bitlen(baseBits, ia.y));
  return (int2) (a, b);
}

// Carry propagation.
int2 car1(long *carry, int2 r, int pos) {
  int a = updateMul(1, carry, r.x, bitlenAtPos(pos));
  int b = updateMul(1, carry, r.y, bitlenAtPos(pos + 1));
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
                  G double2 *io, G double *carryShuttle, volatile global uint *ready,
                  const G double2 *A, const G double2 *iA, const G double2 *trig) {
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

// propagate carry this many lines.
#define CARRY_LEN 16

// Carry propagation with optional MUL, over 16 words.
// Input is doubles. They are weighted with the "inverse weight" A
// and rounded to output ints and to left-over carryOut.
// Width = N * 256
void carryACore(uint N, uint mul, const G double2 *in, const G double2 *A, G int2 *out, G long *carryOut) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);

  uint step = g % N * 256 + g / N * (N * CARRY_LEN * 256);
  in     += step;
  A      += step;
  out    += step;

  long carry = 0;

  for (int i = 0; i < CARRY_LEN; ++i) {
    uint p = me + i * N * 256;
    out[p] = car0Mul(mul, &carry, conjugate(in[p]), A[p], BASE_BITLEN);
  }
  carryOut[g * 256 + me] = carry;
}

// The second round of carry propagation (16 words), needed to "link the chain" after carryA.
// Input is int words and the left-over carry from carryA.
// Output is int words.
// Width = N * 256
void carryBCore(uint N, uint H, G int2 *io, const G long *carryIn) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);
  uint gx = g % N;
  uint gy = g / N;
  
  uint step = g % N * 256 + g / N * (N * CARRY_LEN * 256);
  io += step;

  uint HB = H / CARRY_LEN;
  
  uint prev = (gy + HB * 256 * gx + HB * me - 1) & (HB * N * 256 - 1);
  uint prevLine = prev % HB;
  uint prevCol  = prev / HB;
  long carry = carryIn[N * 256 * prevLine + prevCol];
  
  for (int i = 0; i < CARRY_LEN; ++i) {
    uint pos = CARRY_LEN * gy + H * 256 * gx + H * me + i;
    uint p = me + i * N * 256;
    io[p] = car1(&carry, io[p], pos * 2);
    if (!carry) { return; }
  }
}

double2 foo2(double2 a, double2 b) {
  a = addsub(a);
  b = addsub(b);
  return addsub(a * b);
}

// computes 2*[x^2+y^2 + i*(2*x*y)]. Needs a name.
double2 foo(double2 a) { return foo2(a, a); }

// Inputs normal (non-conjugate); outputs conjugate.
void csquare(uint W, uint H, G double2 *io, const G double2 *bigTrig) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);

  if (g == 0 && me == 0) {
    io[0]     = 4 * foo(conjugate(io[0]));
    io[W / 2] = 8 * sq(conjugate(io[W / 2]));
    return;
  }
  
  uint line = g / (W / 512);
  uint posInLine = g % (W / 512) * 256 + me;
  uint k = line * W + posInLine;
  uint v = ((H - line) & (H - 1)) * W + (W - 1) - posInLine + ((line - 1) >> 31);
  
  double2 a = io[k];
  double2 b = conjugate(io[v]);
  double2 t = swap(mul(bigTrig[W * 2 + H / 2 + line], bigTrig[posInLine]));
  
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
void cmul(uint W, uint H, G double2 *io, const G double2 *in, const G double2 *bigTrig) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);

  if (g == 0 && me == 0) {
    io[0]    = 4 * foo2(conjugate(io[0]), conjugate(in[0]));
    io[W / 2] = 8 * conjugate(mul(io[W / 2], in[W / 2]));
    return;
  }
  
  uint line = g / (W / 512);
  uint posInLine = g % (W / 512) * 256 + me;
  uint k = line * W + posInLine;
  uint v = ((H - line) & (H - 1)) * W + (W - 1) - posInLine + ((line - 1) >> 31);
  
  double2 a = io[k];
  double2 b = conjugate(io[v]);
  double2 t = swap(mul(bigTrig[W * 2 + H / 2 + line], bigTrig[posInLine]));
  
  X2(a, b);
  b = mul(b, conjugate(t));
  X2(a, b);
  
  double2 c = in[k];
  double2 d = conjugate(in[v]);
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

// M == max(W, H)
void transpose(uint W, uint H, uint M, local double *lds, const G double2 *in, G double2 *out, const G double2 *bigTrig) {
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
    u[i] = mul(u[i], bigTrig[M * 2 + k % (W * H / (M * 2))]);
    u[i] = mul(u[i], bigTrig[k / (W * H / (M * 2))]);

    uint p = (my + i * 4) * H + mx;
    out[p] = u[i];
  }
}

void halfSq(uint N, double2 *u, double2 *v, double2 tt, const G double2 *bigTrig, bool special) {
  uint g = get_group_id(0);
  uint me = get_local_id(0);
  for (int i = 0; i < N / 2; ++i) {
    double2 a = u[i];
    double2 b = conjugate(v[N / 2 + i]);
    double2 t = swap(mul(tt, bigTrig[256 * i + me]));  // assert(N == 8); !! 
    if (special && i == 0 && g == 0 && me == 0) {
      a = 4 * foo(a);
      b = 8 * sq(b);
    } else {
      X2(a, b);
      b = mul(b, conjugate(t));
      X2(a, b);
      a = sq(a);
      b = sq(b);
      X2(a, b);
      b = mul(b,  t);
      X2(a, b);
    }
    u[i] = conjugate(a);
    v[N / 2 + i] = b;
  }
}

void convolution(uint N, uint H, local double *lds, double2 *u, double2 *v, G double2 *io, const G double2 *trig, const G double2 *bigTrig) {
  uint W = N * 256;
  uint g = get_group_id(0);
  uint me = get_local_id(0);
  read(N, u, io, g * W);
  fftImpl(N, lds, u, trig);
  reverse(N, (local double2 *) lds, u, g == 0);

  uint line2 = g ? H - g : (H / 2);
  read(N, v, io, line2 * W);
  bar();
  fftImpl(N, lds, v, trig);
  reverse(N, (local double2 *) lds, v, false);
  
  if (g == 0) { for (int i = N / 2; i < N; ++i) { S2(u[i], v[i]); } }

  halfSq(N, u, v, bigTrig[W * 2 + (H / 2) + g],      bigTrig, true);
  
  halfSq(N, v, u, bigTrig[W * 2 + (H / 2) +  line2], bigTrig, false);

  if (g == 0) { for (int i = N / 2; i < N; ++i) { S2(u[i], v[i]); } }

  reverse(N, (local double2 *) lds, u, g == 0);
  bar();
  fftImpl(N, lds, u, trig);
  write(N, u, io, g * W);
  
  reverse(N, (local double2 *) lds, v, false);
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
typedef CP(double2) Trig;

#else

#define P(x) global x *
#define CP(x) const P(x)
typedef CP(double2) restrict Trig;

#endif

KERNEL(256) fftW(P(double2) io, Trig smallTrig) {
  local double lds[WIDTH];
  double2 u[N_WIDTH];
  fft(N_WIDTH, lds, u, io, smallTrig);
}

KERNEL(256) fftH(P(double2) io, Trig smallTrig) {
  local double lds[HEIGHT];
  double2 u[N_HEIGHT];
  fft(N_HEIGHT, lds, u, io, smallTrig);
}

KERNEL(256) fftP(CP(int2) in, P(double2) out, CP(double2) A, Trig smallTrig) {
  local double lds[WIDTH];
  double2 u[N_WIDTH];
  fftPremul(N_WIDTH, lds, u, in, out, A, smallTrig);
}

KERNEL(256) carryA(CP(double2) in, CP(double2) A, P(int2) out, P(long) carryOut) {
  carryACore(N_WIDTH, 1, in, A, out, carryOut);
}

KERNEL(256) carryM(CP(double2) in, CP(double2) A, P(int2) out, P(long) carryOut) {
  carryACore(N_WIDTH, 3, in, A, out, carryOut);
}

KERNEL(256) carryB(P(int2) io, CP(long) carryIn) {
  carryBCore(N_WIDTH, HEIGHT, io, carryIn);
}

KERNEL(256) square(P(double2) io, Trig bigTrig)  { csquare(HEIGHT, WIDTH, io, bigTrig); }

KERNEL(256) multiply(P(double2) io, CP(double2) in, Trig bigTrig)  { cmul(HEIGHT, WIDTH, io, in, bigTrig); }

KERNEL(256) carryConv(P(double2) io, P(double) carryShuttle, volatile P(uint) ready,
                      CP(double2) A, CP(double2) iA, Trig smallTrig) {
  local double lds[WIDTH];
  double2 u[N_WIDTH];
  carryConvolution(N_WIDTH, HEIGHT, lds, u, BASE_BITLEN, io, carryShuttle, ready, A, iA, smallTrig);
}

KERNEL(256) tail(P(double2) io, Trig smallTrig, Trig bigTrig) {
  local double lds[HEIGHT];
  double2 u[N_HEIGHT];
  double2 v[N_HEIGHT];
  convolution(N_HEIGHT, WIDTH, lds, u, v, io, smallTrig, bigTrig);
}

KERNEL(256) transposeW(CP(double2) in, P(double2) out, Trig bigTrig) {
  local double lds[4096];
  transpose(WIDTH, HEIGHT, max(WIDTH, HEIGHT), lds, in, out, bigTrig);
}

KERNEL(256) transposeH(CP(double2) in, P(double2) out, Trig bigTrig) {
  local double lds[4096];
  transpose(HEIGHT, WIDTH, max(WIDTH, HEIGHT), lds, in, out, bigTrig);
}

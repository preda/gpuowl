// gpuOwl, an OpenCL Mersenne primality test.
// Copyright (C) 2017 Mihai Preda.

// The data is organized in pairs of words in a matrix WIDTH x HEIGHT.
// The pair (a, b) is sometimes interpreted as the complex value a + i*b.
// The order of words is column-major (i.e. transposed from the usual row-major matrix order).

// Expected defines: WIDTH, HEIGHT, EXP.

// Number of words
#define NWORDS (625 * 4096 * 2u)

// Used in bitlen() and weighting.
#define STEP (NWORDS - (EXP % NWORDS))

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

uint lo(ulong a) { return a & 0xffffffffu; }
uint up(ulong a) { return a >> 32; }

#pragma OPENCL FP_CONTRACT ON

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double T;
typedef double2 T2;

typedef int Word;
typedef int2 Word2;
typedef long Carry;

T2 U2(T a, T b) { return (T2)(a, b); }

T add1(T a, T b) { return a + b; }
T sub1(T a, T b) { return a - b; }

T2 add(T2 a, T2 b) { return a + b; }
T2 sub(T2 a, T2 b) { return a - b; }

T shl1(T a, uint k) { return a * (1 << k); }

// complex mul
T2 mul(T2 a, T2 b) { return U2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
// T2 mul(T2 a, T2 b) { return U2(a.x * b.x - a.y * b.y, (a.x + a.y) * (b.x + b.y) - a.x * b.x - a.y * b.y); }

// complex square
T2 sq(T2 a) { return U2((a.x + a.y) * (a.x - a.y), 2 * a.x * a.y); }

T mul1(T a, T b) { return a * b; }

T2 mul_t4(T2 a)  { return U2(a.y, -a.x); }                          // mul(a, U2( 0, -1)); }
T2 mul_t8(T2 a)  { return U2(a.y + a.x, a.y - a.x) * M_SQRT1_2; }   // mul(a, U2( 1, -1)) * (T)(M_SQRT1_2); }
T2 mul_3t8(T2 a) { return U2(a.x - a.y, a.x + a.y) * - M_SQRT1_2; } // mul(a, U2(-1, -1)) * (T)(M_SQRT1_2); }


T2 shl(T2 a, uint k) { return U2(shl1(a.x, k), shl1(a.y, k)); }

T2 addsub(T2 a) { return U2(add1(a.x, a.y), sub1(a.x, a.y)); }
T2 swap(T2 a) { return U2(a.y, a.x); }
T2 conjugate(T2 a) { return U2(a.x, -a.y); }

uint extra(ulong k) { return k * STEP % NWORDS; }

void bar()    { barrier(CLK_LOCAL_MEM_FENCE); }
void bigBar() { barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); }

// Is the word at pos a big word (BASE_BITLEN+1 bits)? (vs. a small, BASE_BITLEN bits word).
bool isBigWord(uint k) { return extra(k) + STEP < NWORDS; }

// Number of bits for the word at pos.
uint bitlen(uint k) { return EXP / NWORDS + isBigWord(k); }

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

T2 carryAndWeight(Word2 u, Carry *carry, T2 weight) {
  Word x = carryStep(u.x, carry, oldBitlen(weight.x));
  Word y = carryStep(u.y, carry, oldBitlen(weight.y));
  return weightAux(x, y, weight);
}

// No carry out. The final carry is "absorbed" in the last word.
T2 carryAndWeightFinal(Word2 u, Carry carry, T2 w) {
  Word x = carryStep(u.x, &carry, oldBitlen(w.x));
  Word y = u.y + carry;
  return weightAux(x, y, w);
}

// Generic code below.

// Carry propagation from word and carry.
Word2 carryWord(Word2 a, Carry *carry, uint pos) {
  a.x = carryStep(a.x, carry, bitlen(2 * pos + 0));
  a.y = carryStep(a.y, carry, bitlen(2 * pos + 1));
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

void fft8(T2 *u) {
  for (int i = 0; i < 4; ++i) { X2(u[i], u[i + 4]); }
  u[5] = mul_t8(u[5]);
  u[6] = mul_t4(u[6]);
  u[7] = mul_3t8(u[7]);
  
  fft4Core(u);
  fft4Core(u + 4);

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
  
  X2(u[1], u[4]);
  X2(u[2], u[3]);
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

void tabMul(uint WG, const G T2 *trig, T2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  for (int i = 1; i < n; ++i) { u[i] = mul(u[i], trig[me / f + i * (WG / f)]); }
}

// WG:125, LDS:625*8, u:5.
void fft625Impl(local T *lds, T2 *u, const G T2 *trig) {
  for (int s = 25; s >= 1; s /= 5) {
    fft5(u);
    if (s != 25) { bar(); }
    shufl (125, lds,  u, 5, s);
    tabMul(125, trig, u, 5, s);
  }
  fft5(u);
}

// WG:512, LDS:32KB, u:8.
void fft4KImpl(local T *lds, T2 *u, const G T2 *trig) {
  for (int s = 6; s >= 0; s -= 3) {
    fft8(u);

    if (s != 6) { bar(); }
    shufl (512, lds,  u, 8, 1 << s);
    tabMul(512, trig, u, 8, 1 << s);
  }
  fft8(u);
}

void read(uint WG, uint N, T2 *u, G T2 *in, uint base) {
  for (int i = 0; i < N; ++i) { u[i] = in[base + i * WG + (uint) get_local_id(0)]; }
}

void write(uint WG, uint N, T2 *u, G T2 *out, uint base) {
  for (int i = 0; i < N; ++i) { out[base + i * WG + (uint) get_local_id(0)] = u[i]; }
}

// Carry propagation with optional MUL-3, over CARRY_LEN words.
// Input is conjugated and inverse-weighted.
void carryACore(uint WG, uint N, uint mul, const G T2 *in, const G T2 *A, G Word2 *out, G Carry *carryOut) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);
  uint gx = g % N;
  uint gy = g / N;

  uint line = CARRY_LEN * gy;
  // uint step = WG * gx + N * WG * CARRY_LEN * gy;
  in  += WG * gx + 640 * line;
  out += WG * gx + 625 * line;
  A   += WG * gx + 625 * line;

  Carry carry = 0;

  for (int i = 0; i < CARRY_LEN; ++i) {
    uint p = i * N * WG + me;
    out[p] = unweightAndCarry(mul, conjugate(in[640 * i + me]), &carry, A[p]);
  }
  carryOut[g * WG + me] = carry;
}

// The second round of carry propagation (16 words), needed to "link the chain" after carryA.
void carryBCore(uint WG, uint N, uint H, G Word2 *io, const G Carry *carryIn) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);
  uint gx = g % N;
  uint gy = g / N;
  
  uint step = WG * gx + N * WG * CARRY_LEN * gy;
  io += step;

  uint HB = H / CARRY_LEN;
  
  uint prev = (gy + HB * WG * gx + HB * me + (HB * N * WG - 1)) % (HB * N * WG);
  uint prevLine = prev % HB;
  uint prevCol  = prev / HB;
  Carry carry = carryIn[N * WG * prevLine + prevCol];
  
  for (int i = 0; i < CARRY_LEN; ++i) {
    uint pos = CARRY_LEN * gy + H * WG * gx + H * me + i;
    uint p = i * N * WG + me;
    io[p] = carryWord(io[p], &carry, pos);
    if (!carry) { return; }
  }
}

// Inputs normal (non-conjugate); outputs conjugate.
// bigTrig: see genSquareTrig() in gpuowl.cpp
void csquare(uint WG, uint W, uint H, G T2 *io, const G T2 *bigTrig) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);

  if (g == 0 && me == 0) {
    io[0]     = shl(foo(conjugate(io[0])), 2);
    io[W / 2] = shl(sq(conjugate(io[W / 2])), 3);
    return;
  }

  uint GPL = W / (WG * 2); // "Groups Per Line", == 4.
  uint line = g / GPL;
  uint posInLine = g % GPL * WG + me;

  T2 t = swap(mul(bigTrig[posInLine], bigTrig[4096 + line]));
  
  uint k = line * W + posInLine;
  uint v = ((H - line) % H) * W + (W - 1) - posInLine + ((line - 1) >> 31);
  
  T2 a = io[k];
  T2 b = conjugate(io[v]);
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
void cmul(uint WG, uint W, uint H, G T2 *io, const G T2 *in, const G T2 *bigTrig) {
  // const G T2 *bigTrig) {
  uint g  = get_group_id(0);
  uint me = get_local_id(0);

  if (g == 0 && me == 0) {
    io[0]     = shl(foo2(conjugate(io[0]), conjugate(in[0])), 2);
    io[W / 2] = shl(conjugate(mul(io[W / 2], in[W / 2])), 3);
    return;
  }

  uint GPL = W / (WG * 2);
  uint line = g / GPL;
  uint posInLine = g % GPL * WG + me;

  T2 t = swap(mul(bigTrig[posInLine], bigTrig[4096 + line]));
  
  uint k = line * W + posInLine;
  uint v = ((H - line) % H) * W + (W - 1) - posInLine + ((line - 1) >> 31);
  
  T2 a = io[k];
  T2 b = conjugate(io[v]);
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

// transpose LDS 64 x 64.
void transposeLDS(local T *lds, T2 *u) {
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

void transpose(uint W, uint H, local T *lds, const G T2 *in, G T2 *out, const G T2 *trig) {
  uint GPW = (W - 1) / 64 + 1, GPH = (H - 1) / 64 + 1;
  uint PW = GPW * 64, PH = GPH * 64; // padded to multiple of 64.
  uint g = get_group_id(0), gx = g % GPW, gy = g / GPW;
  gy = (gy + gx) % GPH;

  in   += gy * 64 * PW + gx * 64;
  out  += gy * 64      + gx * 64 * PH;
  
  uint me = get_local_id(0), mx = me % 64, my = me / 64;
  T2 u[16];

  if ((W % 64 == 0) || (gx * 64 + mx < W)) {
    for (int i = 0; i < 16; ++i) {
      if (!(H % 64) || (gy * 64 + i * 4 + my < H)) {
        u[i] = in[(i * 4 + my) * PW + mx];
      }
    }
  }

  transposeLDS(lds, u);

  if (!(H % 64) || (gy * 64 + mx < H)) {  
    for (int i = 0; i < 16; ++i) {
      if ((W % 64 == 0) || (i * 4 + gx * 64 + 3 < W) || (i * 4 + gx * 64 + my < W)) {
        uint k = mul24(gy * 64 + mx, gx * 64 + my + (uint) i * 4);
        u[i] = mul(u[i], mul(trig[k / (W * H / 2048)], trig[2048 + k % (W * H / 2048)]));
        out[(i * 4 + my) * PH + mx] = u[i];
      }
    }
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

KERNEL(125) fft625(P(T2) io, Trig smallTrig) {
  local T lds[5 * 125];
  T2 u[5];

  uint g = get_group_id(0);
  io += 640 * g;

  read(125, 5, u, io, 0);
  fft625Impl(lds, u, smallTrig);
  write(125, 5, u, io, 0);
}

KERNEL(512) fft4K(P(T2) io, Trig smallTrig) {
  local T lds[8 * 512];
  T2 u[8];

  uint g = get_group_id(0);
  io += 4096 * g;

  read(512, 8, u, io, 0);
  fft4KImpl(lds, u, smallTrig);
  write(512, 8, u, io, 0);
}

// fftPremul: weight words with "A" (for IBDWT) followed by FFT.
KERNEL(125) fftP(CP(Word2) in, P(T2) out, CP(T2) A, Trig smallTrig) {
  local T lds[5 * 125];
  T2 u[5];

  uint g = get_group_id(0);
  A   += 625 * g;
  in  += 625 * g;
  out += 640 * g;

  uint me = get_local_id(0);

  for (int i = 0; i < 5; ++i) {
    uint p = 125 * i + me;
    u[i] = weight(in[p], A[p]);
  }

  fft625Impl(lds, u, smallTrig);

  write(125, 5, u, out, 0);
}

KERNEL(125) carryA(CP(T2) in, CP(T2) A, P(Word2) out, P(Carry) carryOut) {
  carryACore(125, 5, 1, in, A, out, carryOut);
}

KERNEL(125) carryM(CP(T2) in, CP(T2) A, P(Word2) out, P(Carry) carryOut) {
  carryACore(125, 5, 3, in, A, out, carryOut);
}

KERNEL(125) carryB(P(Word2) io, CP(Carry) carryIn) {
  carryBCore(125, 5, 4096, io, carryIn);
}

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway" carry data forwarding from one group to the next.

#ifdef CARRY_MEDIUM

KERNEL(125) carryFused(P(T2) io, P(Carry) carryShuttle, volatile P(uint) ready,
                      CP(T2) A, CP(T2) iA, Trig smallTrig) {
  local T lds[625];

  uint gr = get_group_id(0);
  uint me = get_local_id(0);
  uint line = gr % 2048 * 2;

  io += 640 * line;
  A  += 625 * line;
  iA += 625 * line;
  
  T2 u[5]; read(125, 5, u, io, 0);
  fft625Impl(lds, u, smallTrig);
  bar();
  
  T2 v[5]; read(125, 5, v, io, 640);
  fft625Impl(lds, v, smallTrig);
  
  Word2 wu[5], wv[5];
  for (int i = 0; i < 5; ++i) {
    uint p = i * 125 + me;
    Carry carry = 0;
    wu[i] = unweightAndCarry(1, conjugate(u[i]), &carry, iA[p]);
    wv[i] = unweightAndCarry(1, conjugate(v[i]), &carry, iA[p + 625]);
    if (gr < 2048) { carryShuttle[gr * 625 + p] = carry; }
  }

  bigBar();

  // Signal that this group is done writing the carry.
  if (gr < 2048 && me == 0) { atomic_xchg(&ready[gr], 1); }

  if (gr == 0) { return; }

  // Wait until the previous group is ready with the carry.
  if (me == 0) { while(!atomic_xchg(&ready[gr - 1], 0)); }

  bigBar();
  
  for (int i = 0; i < 5; ++i) {
    uint p = i * 125 + me;
    Carry carry = carryShuttle[(gr - 1) * 625 + ((p + 625 - gr / 2048) % 625)];
    u[i] = carryAndWeight(wu[i], &carry, A[p]);
    v[i] = carryAndWeightFinal(wv[i], carry, A[p + 625]);
  }

  fft625Impl(lds, u, smallTrig);
  bar();
  write(125, 5, u, io, 0);
  fft625Impl(lds, v, smallTrig);
  write(125, 5, v, io, 640);
}

#else
// i.e. -carry short
KERNEL(125) carryFused(P(T2) io, P(Carry) carryShuttle, volatile P(uint) ready,
                      CP(T2) A, CP(T2) iA, Trig smallTrig) {
  local T lds[625];

  uint gr = get_group_id(0);
  uint me = get_local_id(0);
  uint line = gr % 4096;

  io += 640 * line;
  A  += 625 * line;
  iA += 625 * line;
  
  T2 u[5];
  read(125, 5, u, io, 0);
  fft625Impl(lds, u, smallTrig);

  Word2 word[5];
  for (int i = 0; i < 5; ++i) {
    uint p = i * 125 + me;
    Carry carry = 0;
    word[i] = unweightAndCarry(1, conjugate(u[i]), &carry, iA[p]);
    if (gr < 4096) { carryShuttle[gr * 625 + p] = carry; }
  }

  bigBar();

  // Signal that this group is done writing the carry.
  if (gr < 4096 && me == 0) { atomic_xchg(&ready[gr], 1); }

  if (gr == 0) { return; }

  // Wait until the previous group is ready with the carry.
  if (me == 0) { while(!atomic_xchg(&ready[gr - 1], 0)); }

  bigBar();
  
  for (int i = 0; i < 5; ++i) {
    uint p = i * 125 + me;
    Carry carry = carryShuttle[(gr - 1) * 625 + ((p + 625 - gr / 4096) % 625)];
    u[i] = carryAndWeightFinal(word[i], carry, A[p]);
  }

  fft625Impl(lds, u, smallTrig);
  write(125, 5, u, io, 0);
}

#endif

KERNEL(256) transposeW(CP(T2) in, P(T2) out, Trig trig) {
  local T lds[4096];
  transpose(625, 4096, lds, in, out, trig);
}

KERNEL(256) transposeH(CP(T2) in, P(T2) out, Trig trig) {
  local T lds[4096];
  transpose(4096, 625, lds, in, out, trig);
}

KERNEL(512) square(P(T2) io, Trig bigTrig)  { csquare(512, 4096, 625, io, bigTrig); }

KERNEL(512) multiply(P(T2) io, CP(T2) in, Trig bigTrig)  { cmul(512, 4096, 625, io, in, bigTrig); }

void reverse8(uint WG, local T2 *lds, T2 *u, bool bump) {
  uint me = get_local_id(0);
  uint rm = WG - 1 - me + bump;
  
  bar();

  lds[rm + 0 * WG] = u[7];
  lds[rm + 1 * WG] = u[6];
  lds[rm + 2 * WG] = u[5];  
  lds[bump ? ((rm + 3 * WG) % (4 * WG)) : (rm + 3 * WG)] = u[4];
  
  bar();
  for (int i = 0; i < 4; ++i) { u[4 + i] = lds[i * WG + me]; }
}

void halfSq(uint WG, uint N, T2 *u, T2 *v, T2 tt, const G T2 *bigTrig, bool special) {
  uint me = get_local_id(0);
  for (int i = 0; i < N / 2; ++i) {
    T2 a = u[i];
    T2 b = conjugate(v[N / 2 + i]);
    T2 t = swap(mul(tt, bigTrig[WG * i + me]));
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
    v[N / 2 + i] = b;
  }
}

// "auto convolution" is equivalent to the sequence: fftH, square, fftH.
KERNEL(512) autoConv(P(T2) io, Trig smallTrig, P(T2) bigTrig) {
  local T lds[4096];
  T2 u[8];
  T2 v[8];

  uint H = 625;
  uint W = 4096;
  uint g = get_group_id(0);
  uint me = get_local_id(0);
  
  read(512, 8, u, io, g * W);
  fft4KImpl(lds, u, smallTrig);

  if (g == 0) {
    reverse8(512, (local T2 *) lds, u, true);
    halfSq(512, 8, u, u, bigTrig[4096], bigTrig, true);
    reverse8(512, (local T2 *) lds, u, true);
  } else {
    reverse8(512, (local T2 *) lds, u, false);
  
    uint line2 = H - g;
    read(512, 8, v, io, line2 * W);
    bar();
    fft4KImpl(lds, v, smallTrig);
    reverse8(512, (local T2 *) lds, v, false);
  
    halfSq(512, 8, u, v, bigTrig[4096 + g],     bigTrig, false);  
    halfSq(512, 8, v, u, bigTrig[4096 + line2], bigTrig, false);

    reverse8(512, (local T2 *) lds, u, g == 0);
    reverse8(512, (local T2 *) lds, v, false);

    bar();
    fft4KImpl(lds, v, smallTrig);
    write(512, 8, v, io, line2 * W);  
  }

  bar();
  fft4KImpl(lds, u, smallTrig);
  write(512, 8, u, io, g * W);  
}

// Copyright Mihai Preda and George Woltman.

#define STR(x) XSTR(x)
#define XSTR(x) #x

#define OVL __attribute__((overloadable))
#define VECTOR(n) __attribute__((ext_vector_type(n)))

#if 0  // We don't use FP
#pragma OPENCL FP_CONTRACT ON
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
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

// Expected defines: EXP the exponent.
// WIDTH, SMALL_HEIGHT, MIDDLE.

#if MIDDLE != 1
#error Only MIDDLE==1 is implemented
#endif
    
#define BIG_HEIGHT (SMALL_HEIGHT * MIDDLE)
#define N (WIDTH * BIG_HEIGHT)
// #define NWORDS (ND * 2u)

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


// __attribute__((opencl_unroll_hint(1))) AKA #pragma unroll(1)

void bar() { barrier(0); }

typedef int i32;
typedef uint u32;
typedef long i64;
typedef ulong u64;
typedef long long i128;
typedef unsigned long long u128;

typedef i32 Word;
typedef u64 Carry;
typedef u64 T;
typedef u64 Weight;

u32  U32(u32 x)   { return x; }
u64  U64(u64 x)   { return x; }
u128 U128(u128 x) { return x; }
i32  I32(i32 x)   { return x; }
i64  I64(i64 x)   { return x; }
i128 I128(i128 x) { return x; }

u32 hiU32(u64 x) { return x >> 32; }
u64 hiU64(u128 x) { return x >> 64; }

#define PRIME 0xffffffff00000001

typedef long long i128;
typedef unsigned long long u128;

u32  U32(u32 x)   { return x; }
u64  U64(u64 x)   { return x; }
u128 U128(u128 x) { return x; }
i32  I32(i32 x)   { return x; }
i64  I64(i64 x)   { return x; }
i128 I128(i128 x) { return x; }

u32 hiU32(u64 x) { return x >> 32; }
u64 hiU64(u128 x) { return x >> 64; }

#define PRIME 0xffffffff00000001u
// PRIME == 2^64 - 2^32 + 1
// 2^64 % PRIME == 0xffffffff == 2^32 - 1
// 2^96 % PRIME == 0xffffffff'00000000 == PRIME - 1

u64 reduce(u64 a) { return (a >= PRIME) ? a - PRIME : a }

u64 mul64(u32 x) { return (U64(x) << 32) - x; } // x * 0xffffffff
u64 mul96(u32 x) { return neg(x); }

// Add modulo PRIME. 2^64 % PRIME == U32(-1).
u64 add(u64 a, u64 b) {
  u64 s = a + b;
  return reduce(s) + (U32(-1) + (s >= a));
  // return (s < a) ? reduce(s) + U32(-1) : s;
  // return s + (U32(-1) + (s >= a));
}

u64 neg(u64 a) {
  assert(a < PRIME);
  return a ? PRIME - a : a;
}

u64 sub(u64 a, u64 b) {
  // return (a >= b) ? a - b : (PRIME - reduce(b - a));
  u64 d = a - b;
  return (d <= a) ? d : neg(-d);
  // return (d <= a) ? d : (PRIME - reduce(-d));
}

u64 reduce(u128 x) { return add(add(U64(x), mul64(x >> 64)), mul96(x >> 96); }
u64 modmul(u64 a, u64 b) { return reduce(U128(a) * b); }
u64 modsq(u64 a) { return modmul(a, a); }
u64 mul1T4(u64 x) { return reduce(U128(x) << 48); }
u64 mul3T4(u64 x) { return modmul(x, 0xfffeffff00000001ull); } // { return reduce(x * U128(0xfffeffffu) + x); } // 

/*
u64 modmul(u64 a, u64 b) {
  u128 ab = U128(a) * b;
  u64 high = ab >> 64;
  u64 l0 = high << 32;
  u64 low2 = l0 - high;
  bool borrow = low2 > l0;
  u32 h = U32(high >> 32) - borrow;
  
  u64 low = U64(ab) + low2;
  bool carry = low < low2;
  h += carry;
  return modadd(low, mulm1(h));
}
*/

/*
u64 modmul(u64 a, u64 b) {
  u128 ab = U128(a) * b;
  u64 low = U64(ab);
  u64 high = ab >> 64;
  u32 hl = U32(high);
  u32 hh = high >> 32;
  u64 s = modadd(low, mulm1(hl));
  u64 hhm1 = mulm1(hh);
#if 0
  s = modadd(s, hhm1 << 32);
  s = modadd(s, mulm1(hhm1 >> 32));
#else
  hhm1 += s >> 32;
  s = modadd((hhm1 << 32) | U32(s), mulm1(hhm1 >> 32));
#endif
  return s;
}
*/

// ---- Bits ----

bool test(u32 bits, u32 pos) { return (bits >> pos) & 1; }

#define STEP (NWORDS - (EXP % NWORDS))

u32 extraK(u32 k) {
#if NWORDS & (NWORDS - 1) == 0
  return STEP * k % NWORDS;
#else
#error NWORDS is not a power of 2
#endif
}

bool isBigExtra(u32 extra) { return extra < NWORDS - STEP; }

#define SMALL_BITS (EXP / NWORDS)
// #define BIG_BITS (SMALL_BITS + 1)

u32 bitlenIsBig(bool isBig) { return SMALL_BITS + isBig; }
u32 bitlenExtra(u32 extra) { return bitlenIsBig(isBigExtra(extra)); }

u32 bitlenK(u32 k) { return isBigWordExtra(extra(k)); }

u32 incExtra(u32 a, u32 b) {
  assert(a < NWORDS && b < NWORDS);
  u32 s = a + b;
  return (s < NWORDS) ? s : (s - NWORDS);
}

// ---- Carry ----

i32 lowBits(i32 x, u32 n) { return (x << (32 - n)) >> (32 - n); }

Word carryStep(u64 u, i64* outCarry, u32 extra) {
  u = reduce(u);
  u64 midpoint = (PRIME - 1u) >> 1;
  i64 balanced = (u > midpoint) ? -i64(PRIME - u) : u;
  u32 nBits = bitlenExtra(extra);
  assert(nBits < 32);
  
  Word w = lowBits(balanced, nBits);
  *outCarry = (balanced >> nBits) + (w < 0);
  return w;
}


#define X2(a, b) { u64 t = a; a = add(a, b); b = sub(t, b); }

#define SWAP(a, b) { u64 t = a; a = b; b = t; }

void dfft4(u64* u) {
  X2(u[0], u[2]);
  X2(u[1], u[3]);
  u[3] = mul1T4(u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);
  SWAP(u[1], u[2]);
}

void ifft4(u64* u) {
  X2(u[0], u[2]);
  X2(u[1], u[3]);
  u[3] = mul3T4(u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);
  SWAP(u[1], u[2]);
}

void shufl(u32 WG, local long* lds, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  u32 m = me / f;

  for (u32 round = 0; round < 4; ++round) {
    if (round) { bar(); }  
    for (u32 i = 0; i < n; ++i) { lds[(m + i * WG / f) / n * f + m % n * WG + me % f] = as_long4(u[i])[round]; }
    bar();
    for (u32 i = 0; i < n; ++i) { as_long4(u[i])[round] = lds[i * WG + me]; }
  }
}

void shufl2(u32 WG, local long* lds, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);

  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);

  for (u32 round = 0; round < 4; ++round) {
    if (round) { bar(); }
    for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = as_long4(u[i])[round]; }
    bar();
    for (u32 i = 0; i < n; ++i) { as_long4(u[i])[round] = lds[i * WG + me]; }
  }
}

void tabMul(u32 WG, const global T2 *trig, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  for (i32 i = 1; i < n; ++i) {
    u[i] = mul(u[i], trig[me / f + i * (WG / f)]);
  }
}

void shuflAndMul(u32 WG, local long* lds, const global T2 *trig, T2 *u, u32 n, u32 f) {
  shufl(WG, lds, u, n, f);
  tabMul(WG, trig, u, n, f);
}

void shuflAndMul2(u32 WG, local long* lds, const global T2 *trig, T2 *u, u32 n, u32 f) {
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

// Returns e^(-i * tau * k / n), (tau == 2*pi represents a full circle). So k/n is the ratio of a full circle.
// Inverse trigonometric direction is chosen as an FFT convention.

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


#define ATTR(x) __attribute__((x))
#define WGSIZE(n) ATTR(reqd_work_group_size(n, 1, 1))

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

T2 slowTrig_SH(u32 k, u32 kBound)  { return tableTrig(k, SMALL_HEIGHT, kBound, TRIG_SH); }

T2 slowTrig_BH(u32 k, u32 kBound)  { return tableTrig(k, BIG_HEIGHT, kBound, TRIG_BH); }

T2 slowTrig_N(u32 k, u32 kBound) { return tableTrig(k, ND, kBound, TRIG_N); }

#define P(x) global x * restrict
#define CP(x) const P(x)
typedef CP(T2) Trig;

u32 transPos(u32 k, u32 width, u32 height) { return k / height + k % height * width; }

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

Weight fweightStep(u32 i) {
  // 2^(k/4) * 1/2 for k in [1..4)
  const Weight TWO_TO_NTH[] = {0,
                           0x9837f0518db8a96f46ad23182e42f6f6ULL,
                           0xb504f333f9de6484597d89b3754abe9fULL,
                           0xd744fccad69d6af439a68bb9902d3fdeULL,
  };
  assert(NW == 4);
  assert(i && i < NW);
  return TWO_TO_NTH[i * STEP % NW];
}

Weight iweightStep(u32 i) {
  // 2^-(k/4) for k in [1..4)
  const Weight TWO_TO_MINUS_NTH[] = {0,
                                 0xd744fccad69d6af439a68bb9902d3fdeULL,
                                 0xb504f333f9de6484597d89b3754abe9fULL,
                                 0x9837f0518db8a96f46ad23182e42f6f6ULL,
  };
  assert(NW == 4);
  assert(i && i < NW);
  return TWO_TO_MINUS_NTH[i * STEP % NW];
}

/*
Weight fweightUnitStep(u32 i) {
  Weight FWEIGHTS_[] = FWEIGHTS;
  return FWEIGHTS_[i];
}

Weight iweightUnitStep(u32 i) {
  Weight IWEIGHTS_[] = IWEIGHTS;
  return IWEIGHTS_[i];
}
*/

// Do an fft_WIDTH after a fftMiddleOut (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_W) fftW(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[WIDTH / 2];
  
  T2 u[NW];
  u32 g = get_group_id(0);

  readCarryFusedLine(in, u, g);
  
  fft_WIDTH(lds, u, smallTrig);

  Weight w = WEIGHTS[G_W * g + me].x;

  for (u32 i = 0; i < NW; ++i) {
    u[i] = mul(u[i], (Weight2)(w, updateWeight(w, IWEIGHT_STEP)));
    w = updateWeight(w, 0xd744fccad69d6af439a68bb9902d3fdeULL);
  }
  
  out += WIDTH * g;
  write(G_W, NW, u, out, 0);
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

  Weight w = WEIGHTS[G_W * g + me].y;

  for (u32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // u[i].x = mul(shiftUp(in[p].x), w);
    // u[i].y = mul(shiftUp(in[p].y), updateWeight(w, WEIGHT_STEP));
    u[i] = mul(shiftUp(in[p]), (Weight2)(w, updateWeight(w, FWEIGHT_STEP)));
    w = updateWeight(w, 0x9837f0518db8a96f46ad23182e42f6f6ULL);
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

void middleMul(T2 *u, u32 y) {
  assert(y < SMALL_HEIGHT);

#if 0
  T2 w = slowTrig_BH(y, SMALL_HEIGHT);
  T2 step = w;

  if (MIDDLE > 1) {
    u[1] = mul(u[1], w);
    w = sq(w);
  }  
  
  for (u32 i = 2; i < MIDDLE; ++i) {
    u[i] = mul(u[i], w);
    w = mul(w, step);
  }
#endif
  
  for (u32 i = 1; i < MIDDLE; ++i) { u[i] = mul(u[i], slowTrig_BH(y * i, SMALL_HEIGHT * i)); }
}

void middleMul2(T2 *u, u32 x, u32 y) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);

  T2 w = slowTrig_N(x * y, ND / MIDDLE);
  u[0] = mul(u[0], w);
    
  T2 step = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE);
  for (u32 i = 1; i < MIDDLE; ++i) {
    w = mul(w, step);
    u[i] = mul(u[i], w);
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


kernel WGSIZE(G_W) carryOut(P(i32) outWords, P(u64) outCarry, CP(u64) inBase, Trig smallTrig, CP(u64) invWeights) {
  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);
  // u32 gx = gr % (BIG_HEIGHT / 4);
  // u32 gy = gr / (BIG_HEIGHT / 4);

  local u64 lds[WIDTH];

  inBase += 4 * gr;

  /*
  u64 baseWeight = invWeights[G_W * gr + me];
  u32 extraStep1 = extraK(1);
  u32 extraStepGW = extraK(G_W);
  u32 weight = baseWeight;
  */

  u32 baseExtra = extraK(WIDTH * 4 * gr + me);
  
  u64 u[4];
  for (u32 round = 0, extra = baseExtra; round < 4; ++round, extra = incExtra(extra, extraK(G_W))) {
    CP(u64) in = inBase + BIG_HEIGHT * G_W * round;
    for (u32 i = 0; i < 4; ++i) {
      u[i] = in[BIG_HEIGHT * (G_W / 4) * i + BIG_HEIGHT * (me / 4) + me % 4];
    }
    fft_WIDTH(u, lds);

    for (u32 i = 0; i < 4; ++i) {
      u[i] = modmul(u[i], invWeights[WIDTH * gr + G_W * i + me]);
      i32 word = carryStep(u[i], extra);
    }

    
  }

  
  
  // for (u32 i = 0; i < 16; ++i) { lds[me + WG * i] = in[me % 4 + me / 4 * BIG_HEIGHT + gr * 4 + i * (G_W / 4) * BIG_HEIGHT]; }
  // bar();

  inBase += gr * 4;
  

  for (u32 round = 0; round < 4; ++round) {
    // local u64* lds = ldsBase + WIDTH * round;

    

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

void transposeWords(u32 W, u32 H, local Word2 *lds, const Word2 *in, Word2 *out) {
  u32 GPW = W / 16, GPH = H / 16;

  u32 g = get_group_id(0);
  u32 gx = g % GPW;
  u32 gy = g / GPW;
  
  in  += 16 * gx + 16 * W * gy;
  out += 16 * H * gx + 16 * gy;

  u32 me = get_local_id(0);
  u32 mx = me % 16;
  u32 my = me / 16;
    
  lds[me] = in[my * W + mx];
  bar();
  out[my * H + mx] = in[16 * mx + my];

#if 0
  u32 gy = g % GPH;
  u32 gx = g / GPH;
  gx = (gy + gx) % GPW;

  in   += 16 * W * gy + 16 * gx;
  out  += 16 * gy + 16 * H * gx;
  u32 me = get_local_id(0);
  #pragma unroll 1
  for (i32 i = 0; i < 16; ++i) {
    lds[i * 16 + me] = in[i * W + me];
  }
  bar();
  #pragma unroll 1
  for (i32 i = 0; i < 16; ++i) {
    out[i * H + me] = lds[me * 16 + i];
  }
#endif
}

// from transposed to sequential.
KERNEL(256) transposeOut(P(Word2) out, CP(Word2) in) {
  local T2 lds[256];
  transposeWords(WIDTH, BIG_HEIGHT, lds, in, out);
}

// from sequential to transposed.
KERNEL(256) transposeIn(P(Word2) out, CP(Word2) in) {
  local T2 lds[256];
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
\
T2 b2 = sq(b);         \
  b = mulShl(a, b, 1); \
  a = mul(b2, conjugate_t_squared) + sq(a); \
\
X2conja(a, b);                                  \
}

#if 0
#define onePairSq(a, b, t2) {
  X2conjb(a, b);             

T2 b2 = sq(b);
T2 a2 = sq(a);
b = mulShl(a, b, 1);
a = mul(b2, t2) + a2;

X2conja(a, b);
}
#endif


void pairSq(u32 N, T2 *u, T2 *v, T2 base_squared, bool special) {
  u32 me = get_local_id(0);

  if (special && me == 0) {
    u[0] = foo_m2(conjugate(u[0]));
    v[0] = sqShl(conjugate(v[0]), 2);
  } else {
    onePairSq(u[0], v[0], -base_squared);
  }

  if (N == NH) { onePairSq(u[2], v[2], base_squared); }

  T2 new_base_squared = mul_t4(base_squared);
  onePairSq(u[1], v[1], -new_base_squared);

  if (N == NH) { onePairSq(u[3], v[3], new_base_squared); }
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
  \
  T2 tmp = mad(a, c, mul(mul(b, d), conjugate_t_squared)); \
  b = mad(b, c, mul(a, d)); \
  a = tmp; \
  \
  X2conja(a, b); \
}

void pairMul(u32 N, T2 *u, T2 *v, T2 *p, T2 *q, T2 base_squared, bool special) {
  u32 me = get_local_id(0);

  if (special && me == 0) {
    u[0] = conjugate(foo2_m2(u[0], p[0]));
    v[0] = mulShl(conjugate(v[0]), conjugate(q[0]), 2);
  } else {
    onePairMul(u[0], v[0], p[0], q[0], -base_squared);
  }

  if (N == NH) { onePairMul(u[2], v[2], p[2], q[2], base_squared); }

  T2 new_base_squared = mul_t4(base_squared);
  onePairMul(u[1], v[1], p[1], q[1], -new_base_squared);

  if (N == NH) { onePairMul(u[3], v[3], p[3], q[3], new_base_squared); }
}

KERNEL(G_H) tailFusedSquare(P(T2) out, CP(T2) in, Trig smallTrig1, Trig smallTrig2) {
  local T2 lds[SMALL_HEIGHT / 2];

  T2 u[NH], v[NH];

  u32 W = SMALL_HEIGHT;
  u32 H = ND / W; // WIDTH * MIDDLE

  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);

  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  
  fft_HEIGHT(lds, u, smallTrig1);
  bar();
  fft_HEIGHT(lds, v, smallTrig1);

  u32 me = get_local_id(0);
  T2 trig = TRIG_TRANS[SMALL_HEIGHT / 4 * line1 + me];
  
  if (line1 == 0) {
    // Line 0 is special: it pairs with itself, offseted by 1.
    reverse(G_H, lds, u + NH/2, true);    
    pairSq(NH/2, u,   u + NH/2, trig, true);
    reverse(G_H, lds, u + NH/2, true);

    // Line H/2 also pairs with itself (but without offset).
    reverse(G_H, lds, v + NH/2, false);
    pairSq(NH/2, v,   v + NH/2, TRIG_TRANS[SMALL_HEIGHT / 4 * (H / 2) + me], false);
    // slowTrig_2SH(1 + 2 * me, SMALL_HEIGHT / 2), false);
    reverse(G_H, lds, v + NH/2, false);
  } else {    
    reverseLine(G_H, lds, v);
    
    pairSq(NH, u, v, trig, false);
    // slowTrig_N(line1 + me * H, ND / 4), false);
    reverseLine(G_H, lds, v);
  }

  bar();
  fft_HEIGHT(lds, v, smallTrig2);
  bar();
  fft_HEIGHT(lds, u, smallTrig2);

  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);
  write(G_H, NH, v, out, memline2 * SMALL_HEIGHT);
  write(G_H, NH, u, out, memline1 * SMALL_HEIGHT);
}

KERNEL(G_H) tailFusedMul(P(T2) out, CP(T2) in, CP(T2) a, Trig smallTrig1, Trig smallTrig2) {
  // The arguments smallTrig1, smallTrig2 point to the same data; they are passed in as two buffers instead of one
  // in order to work-around the ROCm optimizer which would otherwise "cache" the data once read into VGPRs, leading
  // to poor occupancy.
  
  local T2 lds[SMALL_HEIGHT / 2];

  T2 u[NH], v[NH];
  T2 p[NH], q[NH];

  u32 W = SMALL_HEIGHT;
  u32 H = ND / W;

  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);
  
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

  u32 me = get_local_id(0);
  T2 trig = TRIG_TRANS[SMALL_HEIGHT / 4 * line1 + me];
  if (line1 == 0) {
    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, trig, true);
    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);

    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, TRIG_TRANS[SMALL_HEIGHT / 4 * (H / 2) + me], false);
    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
  } else {    
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);
    pairMul(NH, u, v, p, q, trig, false);
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);
  }
  
  bar();
  fft_HEIGHT(lds, v, smallTrig2);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);
  write(G_H, NH, v, out, memline2 * SMALL_HEIGHT);

  bar();
  fft_HEIGHT(lds, u, smallTrig2);
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  write(G_H, NH, u, out, memline1 * SMALL_HEIGHT);
}
 
// Generate a small unused kernel so developers can look at how well individual macros assemble and optimize
#ifdef TEST_KERNEL

kernel void testKernel(global ulong* io) {
  uint me = get_local_id(0);

  ulong a = io[me];
  ulong b = io[me + 1];
  io[me] = modmul(a, b);
}

#endif

// Copyright Mihai Preda and George Woltman.

#define STR(x) XSTR(x)
#define XSTR(x) #x

#define OVL __attribute__((overloadable))

// 64-bit atomics used in kernel sum64
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#if DEBUG
#define assert(condition) if (!(condition)) { printf("assert(%s) failed at line %d\n", STR(condition), __LINE__ - 1); }
// __builtin_trap();
#else
#define assert(condition)
//__builtin_assume(condition)
#endif

#if AMDGPU
// On AMDGPU the default is HAS_ASM
#if !NO_ASM
#define HAS_ASM 1
#endif
#endif

// Expected defines: EXP the exponent.
// WIDTH, HEIGHT.

#define N (WIDTH * HEIGHT)

#if WIDTH == 1024 || WIDTH == 256
#define NW 4
#else
#error WIDTH
#endif

#if HEIGHT == 4096 || HEIGHT == 1024 || HEIGHT == 256
#define NH 4
#else
#error HEIGHT
#endif

#define G_W (WIDTH / NW)
#define G_H (HEIGHT / NH)

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

#define PRIME 0xffffffff00000001u
// PRIME == 2^64 - 2^32 + 1
// 2^64 % PRIME == 0xffffffff == 2^32 - 1
// 2^96 % PRIME == 0xffffffff'00000000 == PRIME - 1

u64 neg(u64 a) {
  assert(a < PRIME);
  return a ? PRIME - a : a;
}

u64 mul64(u32 x) { return (U64(x) << 32) - x; } // x * 0xffffffff
u64 mul96(u32 x) { return neg(x); }

u64 reduce64(u64 a) { return (a >= PRIME) ? a - PRIME : a; }

// Add modulo PRIME. 2^64 % PRIME == U32(-1).
u64 add(u64 a, u64 b) {
  u64 s = a + b;
  return reduce64(s) + (U32(-1) + (s >= a));
  // return (s < a) ? reduce(s) + U32(-1) : s;
  // return s + (U32(-1) + (s >= a));
}

u64 sub(u64 a, u64 b) {
  // return (a >= b) ? a - b : (PRIME - reduce(b - a));
  u64 d = a - b;
  return (d <= a) ? d : neg(-d);
  // return (d <= a) ? d : (PRIME - reduce(-d));
}


u64 reduce128(u128 x) { return add(add(U64(x), mul64(x >> 64)), mul96(x >> 96)); }

u64 mul(u64 a, u64 b) { return reduce128(U128(a) * b); }
u64 sq(u64 a) { return mul(a, a); }
u64 mul1T4(u64 x) { return reduce128(U128(x) << 48); }
u64 mul3T4(u64 x) { return mul(x, 0xfffeffff00000001ull); } // { return reduce(x * U128(0xfffeffffu) + x); } // 

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

#define STEP (N - (EXP % N))

u32 extraK(u32 k) {
  
#if (N & (N - 1)) == 0
  return STEP * k % N;
#else
#error N is not a power of 2
#endif

}

u32 incExtra(u32 a, u32 b) {
  assert(a < N && b < N);
  u32 s = a + b;
  return (s < N) ? s : (s - N);
}

bool isBigExtra(u32 extra) { return extra < N - STEP; }

#define SMALL_BITS (EXP / N)

u32 bitlenIsBig(bool isBig) { return SMALL_BITS + isBig; }
u32 bitlenExtra(u32 extra) { return bitlenIsBig(isBigExtra(extra)); }

u32 bitlenK(u32 k) { return isBigExtra(extraK(k)); }

// ---- Carry ----

i32 lowBits(i32 x, u32 n) { return (x << (32 - n)) >> (32 - n); }

#define MIDPOINT ((PRIME - 1u) >> 1u)
i64 balance(u64 x) { return (x > MIDPOINT) ? -I64(PRIME - x) : x; };
u64 unbalance(i64 x) { return (x < 0) ? x + PRIME : x; }
#undef MIDPOINT

Word doCarry(i64 balanced, i64* outCarry, u32 extra) {
  u32 nBits = bitlenExtra(extra);
  assert(nBits < 32);
  Word w = lowBits(balanced, nBits);
  *outCarry = (balanced >> nBits) + (w < 0);
  return w;
}

Word carryStep(u64 u, i64* inOutCarry, u32 extra, u64 iWeight) {
  assert(u < PRIME);
  u = mul(u, iWeight);
  return doCarry(balance(u) + *inOutCarry, inOutCarry, extra);
}

u64 carryWord(i64 w, i32* outCarry, u32 extra, u64 dWeight) {
  i64 carryAux;
  w = doCarry(w, &carryAux, extra);
  *outCarry = carryAux;
  return mul(unbalance(w), dWeight);
}

u64 carryFinal(i32 w, u64 dWeight) {
  return mul(unbalance(w), dWeight);
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

void shufl(u32 me, u32 WG, local u64* lds, u64* u, u32 n, u32 f) {
  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);
  
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i]; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i] = lds[i * WG + me]; }
}

typedef constant const u64* Trig;
 
void tabMul(u32 me, u32 WG, Trig trig, u64* u, u32 n, u32 f) {
  for (u32 i = 1; i < n; ++i) { u[i] = mul(u[i], trig[(me & ~(f-1)) + (i - 1) * WG]); }
}

void shuflAndMul(u32 me, u32 WG, local u64* lds, Trig trig, u64* u, u32 n, u32 f) {
  tabMul(me, WG, trig, u, n, f);
  shufl(me, WG, lds, u, n, f);
}

// 256x4
void dFFT1K(u32 me, local u64* lds, u64* u, Trig trig) {
  // UNROLL_WIDTH_CONTROL
  for (i32 s = 0; s <= 6; s += 2) {
    if (s) { bar(); }
    dfft4(u);
    shuflAndMul(me, 256, lds, trig, u, 4, 1u << s);
  }
  dfft4(u);
}

void iFFT1K(u32 me, local u64* lds, u64* u, Trig trig) {
  // UNROLL_WIDTH_CONTROL
  for (i32 s = 0; s <= 6; s += 2) {
    if (s) { bar(); }
    ifft4(u);
    shuflAndMul(me, 256, lds, trig, u, 4, 1u << s);
  }
  ifft4(u);
}

// 1024x4
void dFFT4K(u32 me, local u64* lds, u64* u, Trig trig) {
  // UNROLL_WIDTH_CONTROL
  for (i32 s = 0; s <= 8; s += 2) {
    if (s) { bar(); }
    dfft4(u);
    shuflAndMul(me, 1024, lds, trig, u, 4, 1u << s);
  }
  dfft4(u);
}

void iFFT4K(u32 me, local u64* lds, u64* u, Trig trig) {
  // UNROLL_WIDTH_CONTROL
  for (i32 s = 0; s <= 8; s += 2) {
    if (s) { bar(); }
    ifft4(u);
    shuflAndMul(me, 1024, lds, trig, u, 4, 1u << s);
  }
  ifft4(u);
}

#define ATTR(x) __attribute__((x))
#define WGSIZE(n) ATTR(reqd_work_group_size(n, 1, 1))

#define P(x) global x * restrict
#define CP(x) const P(x)

kernel WGSIZE(WIDTH) void carryIn(P(u64) out, CP(i32) inWords, CP(i64) inCarry, Trig smallTrig, Trig bigTrig, Trig bigTrigStep, CP(u64) dWeights) {
  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 gm1 = (HEIGHT / 4 - 1 + gr) % (HEIGHT / 4);
  u32 mm1 = (WIDTH - 1 + me) % WIDTH;
  i64 carry64 = inCarry[WIDTH * gm1 + (gr ? me : mm1)];
  i32 carry = 0;

  local u64 lds[WIDTH * 4];

  u32 extra = extraK(4 * gr + HEIGHT * me);
  u64 dWeight = dWeights[WIDTH * gr + me];
  
  lds[me] = carryWord(inWords[4 * WIDTH * gr + me] + carry64, &carry, extra, dWeight);
  
  for (int i = 1; i < 4; ++i) {
    bool isBig = isBigExtra(extra);
    dWeight = mul(dWeight, isBig ? DWSTEP : DWSTEP_2);
    extra = isBig ? extra + STEP : (extra + STEP - N);
    i32 w = inWords[4 * WIDTH * gr + WIDTH * i + me];
    lds[i * WIDTH + me] = (i < 3) ? carryWord(w + carry, &carry, extra, dWeight) : carryFinal(w + carry, dWeight);
  }

  bar();
  
  {
    u64 u[4];
    u32 mx = me % 256;
    u32 my = me / 256;
    for (int i = 0; i < 4; ++i) { u[i] = lds[256 * i + mx + WIDTH * my]; }

    dFFT1K(mx, lds + WIDTH * my, u, smallTrig);

    bar();
    
    for (int i = 0; i < 4; ++i) { lds[256 * i + mx + WIDTH * my] = u[i]; }
  }

  bar();

  if (me < 256) {
    u32 mx = me % 4;
    u32 my = me / 4;
    u64 trig = bigTrig[gr * 256 + me];
    u64 trigStep = bigTrigStep[4 * gr + mx];
    for (int i = 0; i < 16; ++i) {
      u64 u = lds[my + WIDTH * mx + 64 * i];
      out[4 * gr + 64 * HEIGHT * i + mx + HEIGHT * my] = mul(u, trig);
      trig = mul(trig, trigStep);
    }
  }  
}

kernel WGSIZE(WIDTH) void carryOut(P(i32) outWords, P(i64) outCarry, CP(u64) in, Trig smallTrig, Trig bigTrig, Trig bigTrigStep, CP(u64) iWeights) {
  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);
  
  local u64 lds[WIDTH * 4];

  if (me < 256) {
    u32 mx = me % 4;
    u32 my = me / 4;
    u64 trig     = bigTrig[256 * gr + me];
    u64 trigStep = bigTrigStep[4 * gr + mx];
    
    for (int i = 0; i < 16; ++i) {
      u64 u = in[4 * gr + 64 * HEIGHT * i + mx + HEIGHT * my];
      lds[my + WIDTH * mx + 64 * i] = mul(u, trig);
      trig = mul(trig, trigStep);
    }
  }
  
  bar();

  {
    u64 u[4];
    u32 mx = me % 256;
    u32 my = me / 256;
    for (int i = 0; i < 4; ++i) { u[i] = lds[256 * i + mx + WIDTH * my]; }

    iFFT1K(mx, lds + WIDTH * my, u, smallTrig);

    bar();

    for (int i = 0; i < 4; ++i) { lds[256 * i + mx + WIDTH * my] = u[i]; }
  }

  bar();

  i64 carry = 0;
  u32 extra = extraK(4 * gr + HEIGHT * me);
  u64 iWeight = iWeights[WIDTH * gr + me];
  
  for (int i = 0; i < 4; ++i) {
    outWords[4 * WIDTH * gr + WIDTH * i + me] = carryStep(lds[i * WIDTH + me], &carry, extra, iWeight);
    bool isBig = isBigExtra(extra);
    iWeight = mul(iWeight, isBig ? IWSTEP : IWSTEP_2);
    extra = isBig ? extra + STEP : (extra + STEP - N);
  }
  outCarry[WIDTH * gr + me] = carry;
}

kernel WGSIZE(1024) void tailSquare(P(u64) io, Trig dSmallTrig, Trig iSmallTrig) {
  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  local u64 lds[HEIGHT];

  u64 u[4];

  for (int i = 0; i < 4; ++i) { u[i] = io[HEIGHT * gr + 1024u * i + me]; }

  dFFT4K(me, lds, u, dSmallTrig);

  bar();
  
  for (int i = 0; i < 4; ++i) { u[i] = sq(u[i]); }

  iFFT4K(me, lds, u, iSmallTrig);

  for (int i = 0; i < 4; ++i) { io[HEIGHT * gr + 1024u * i + me] = u[i]; }
}

kernel WGSIZE(1024) void tailMul(P(u64) io, CP(u64) inB, Trig dSmallTrig, Trig iSmallTrig) {
  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  local u64 lds[HEIGHT];

  u64 u[4], v[4];

  for (int i = 0; i < 4; ++i) { u[i] = io[HEIGHT * gr + 1024u * i + me]; }
  for (int i = 0; i < 4; ++i) { v[i] = inB[HEIGHT * gr + 1024u * i + me]; }

  dFFT4K(me, lds, u, dSmallTrig);

  bar();

  dFFT4K(me, lds, v, dSmallTrig);

  bar();
  
  for (int i = 0; i < 4; ++i) { u[i] = mul(u[i], v[i]); }

  iFFT4K(me, lds, u, iSmallTrig);

  for (int i = 0; i < 4; ++i) { io[HEIGHT * gr + 1024u * i + me] = u[i]; }
}

void trans32(P(u32) out, P(u32) in, u32 width, u32 height, local u32* lds) {
  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 gx = gr % (width / 32);
  u32 gy = gr / (width / 32);
  u32 mx = me % 32;
  u32 my = me / 32;
  
  lds[32 * mx + my] = in[width * (32 * gy + my) + 32 * gx + mx];
  bar();
  out[height * (32 * gx + my) + 32 * gy + mx] = lds[32 * my + mx];
}

void trans64(P(i64) out, P(i64) in, u32 width, u32 height, local i64* lds) {
  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 gx = gr % (width / 32);
  u32 gy = gr / (width / 32);
  u32 mx = me % 32;
  u32 my = me / 32;
  
  lds[32 * mx + my] = in[width * (32 * gy + my) + 32 * gx + mx];
  bar();
  out[height * (32 * gx + my) + 32 * gy + mx] = lds[32 * my + mx];
}

kernel WGSIZE(1024) void transposeWordsOut(P(u32) out, P(u32) in) {
  local u32 lds[1024];
  trans32(out, in, WIDTH, HEIGHT, lds);
}

kernel WGSIZE(1024) void transposeWordsIn(P(u32) out, P(u32) in) {
  local u32 lds[1024];
  trans32(out, in, HEIGHT, WIDTH, lds);
}

kernel WGSIZE(1024) void transposeCarryOut(P(i64) out, P(i64) in) {
  local i64 lds[1024];
  trans64(out, in, WIDTH, HEIGHT / 4, lds);
}

// Generate a small unused kernel so developers can look at how well individual macros assemble and optimize
kernel void testKernel(global ulong* io) {
  uint me = get_local_id(0);

  ulong a = io[me];
  ulong b = io[me + 1];
  io[me] = mul(a, b);
}

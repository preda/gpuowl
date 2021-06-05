// Copyright Mihai Preda and George Woltman.

#define STR(x) XSTR(x)
#define XSTR(x) #x

#define OVL __attribute__((overloadable))

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

/*
#if HEIGHT == 4096
#define NH 8
#else
#error HEIGHT
#endif
*/

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

u32  U32(u32 x)   { return x; }
u64 OVL U64(u64 x)   { return x; }
u64 OVL U64(u32 a, u32 b) { return as_ulong((uint2)(a, b)); }
u128 OVL U128(u128 x) { return x; }
u128 OVL U128(u64 a, u64 b) { return (U128(b) << 64) | a; }
i32  I32(i32 x)   { return x; }
i64  I64(i64 x)   { return x; }
i128 I128(i128 x) { return x; }

u32 hiU32(u64 x) { return x >> 32; }
u64 hiU64(u128 x) { return x >> 64; }

#define PRIME 0xffffffff00000001u
// PRIME == 2^64 - 2^32 + 1
// 2^64 % PRIME == 0xffffffff == 2^32 - 1
// 2^96 % PRIME == 0xffffffff'00000000 == PRIME - 1


u64 reduce64(u64 x) {
  u32 a, b;
  // u32 dummy;
#if HAS_ASM
  __asm("v_add_co_u32_e32 %[a], vcc, -1, %[xLo]\n\t"
        "v_addc_co_u32_e32 %[b], vcc, 0, %[xHi], vcc\n\t"
        "v_cndmask_b32_e32 %[a], %[xLo], %[a], vcc\n\t"
        "v_cndmask_b32_e32 %[b], %[xHi], %[b], vcc\n\t"
        : [a] "=&v"(a), [b] "=&v"(b)
        : [xLo] "v"(U32(x)), [xHi] "v"(U32(x >> 32))// , "{exec}"(dummy)
        : "vcc");
  return U64(a, b);
#else
  return (x >= PRIME) ? x - PRIME : x;
#endif
}

// Add modulo PRIME. 2^64 % PRIME == U32(-1).
u64 add(u64 a, u64 b) {
#if HAS_ASM

  u32 c, d;
  u32 tmp;
  // u32 dummy;
  __asm("#ADD\n\t"
        "v_add_co_u32_e32 %0, vcc, %3, %5\n\t"
        "v_addc_co_u32_e32 %1, vcc, %4, %6, vcc\n\t"
        "v_subbrev_co_u32_e32 %2, vcc, %0, %0, vcc\n\t" // The value of %0 does not matter, it's just a way to make a Zero.
        "v_add_co_u32_e32 %0, vcc, %0, %2\n\t"
        "v_addc_co_u32_e32 %1, vcc, 0, %1, vcc"        
        : "=&v"(c), "=v"(d), "=v"(tmp)
        : "v"(U32(a)), "v"(U32(a>>32)), "v"(U32(b)), "v"(U32(b>>32))// , "{exec}"(dummy)
        : "vcc");
  return U64(c, d);

#else
  
  u64 s = a + b;
  return s + -U32(s < a);
  
#endif
}

// u64 add(u64 a, u64 b) { return reduce64(add1(a, b)); }

u64 sub(u64 a, u64 b) {
#if HAS_ASM
  u32 c, d;
  u64 stmp;
  // u32 dummy;
#if 1
  __asm("#SUB\n\t"
        "v_sub_co_u32_e32  %[c], vcc, %[aLo], %[bLo]\n\t"
	"v_subb_co_u32_e32 %[d], vcc, %[aHi], %[bHi], vcc\n\t"
        "v_addc_co_u32 %[c], %[stmp], 0, %[c], vcc\n\t"
        "s_orn2_b64_e32 vcc, %[stmp], vcc\n\t"
        "v_addc_co_u32_e32 %[d], vcc, -1, %[d], vcc"        
        : [c] "=&v"(c), [d] "=v"(d), [stmp] "=&s"(stmp)
        : [aLo] "v"(U32(a)), [aHi] "v"(U32(a>>32)), [bLo] "v"(U32(b)), [bHi] "v"(U32(b>>32))// , "{exec}"(dummy)
        : "vcc", "scc");
#else
  __asm("#SUB\n\t"
        "v_sub_co_u32_e32  %[c], vcc, %[aLo], %[bLo]\n\t"
	"v_subb_co_u32_e32 %[d], vcc, %[aHi], %[bHi], vcc\n\t"
        "s_mov_b64_e32 %[stmp], vcc\n\t"
        "v_addc_co_u32 %[c], vcc, 0, %[c], vcc\n\t"
        "s_orn2_b64_e32 vcc, vcc, %[stmp]\n\t"
        "v_addc_co_u32_e32 %[d], vcc, -1, %[d], vcc"        
        : [c] "=&v"(c), [d] "=v"(d), [stmp] "=&s"(stmp)
        : [aLo] "v"(U32(a)), [aHi] "v"(U32(a>>32)), [bLo] "v"(U32(b)), [bHi] "v"(U32(b>>32))// , "{exec}"(dummy)
        : "vcc", "scc");
#endif
  return U64(c, d);
#else
  // return a - b - -U32(a < b);
  // return (a >= b) ? a - b : (a - b - 0xffffffff);
  
  u64 d = a - b;
  if (a >= b) { return d; }
  return (d >= 0xffffffff) ? d - 0xffffffff : (d + 0xfffffffe00000002);
#endif
}

// u64 sub(u64 a, u64 b) { return reduce64(sub1(a, b)); }

u64 mul64(u32 x) {
#if HAS_ASM
  u32 a, b;
  // u32 dummy;
  __asm("#SHL64\n\t"
        "v_sub_co_u32_e32 %0, vcc, 0, %2\n\t"
        "v_subbrev_co_u32_e32 %1, vcc, 0, %2, vcc"
        : "=&v"(a), "=v"(b)
        : "v"(x)// , "{exec}"(dummy)
        : "vcc");
  return U64(a, b);
#else
  return (U64(x) << 32) - x; // x * 0xffffffff
#endif
}

u64 mul64w(u64 x) {
#if HAS_ASM
  u32 a, b, c;
  // u32 dummy;
  __asm("#MUL64w\n\t"
        "v_sub_co_u32_e32 %[a], vcc, 0, %[xLo]\n\t"
        "v_subb_co_u32_e32 %[b], vcc, %[xLo], %[xHi], vcc\n\t"
        "v_subbrev_co_u32_e32 %[c], vcc, 0, %[xHi], vcc"
        : [a] "=&v"(a), [b] "=&v"(b), [c] "=v"(c)
        : [xLo] "v" (U32(x)), [xHi] "v" (U32(x >> 32))// , "{exec}"(dummy)
        : "vcc");
  return add(U64(a, b), mul64(c));
#else
  u128 tmp = (U128(x) << 32) - x;
  return add(U64(tmp), mul64(U32(tmp >> 64)));
#endif
}

u64 OVL auxMul(u32 x, u32 y, u64 carry) {
#if HAS_ASM
  u64 out;
  u64 dummy;
  __asm("v_mad_u64_u32 %0, %1, %2, %3, %4"
        : "=v"(out), "=s"(dummy)
        : "v"(x), "v"(y), "v"(carry)// , "{exec}"(dummy)
        );
  return out;
#else
  return U64(x) * y + carry;
#endif
}

u64 OVL auxMul(u32 x, u32 y) {
#if HAS_ASM && 0
  u64 out;
  u64 dummy;
  // __asm("v_mad_u64_u32 %0, %1, %2, %3, 0" : "=v"(out), "=s"(dummy) : "v"(x), "v"(y));
  __asm("v_mad_u64_u32 %0, vcc, %[x], %[y], 0" : "=v"(out) : [x] "v"(x), [y] "v"(y), "{exec}"(dummy) : "vcc");
  return out;
#else
  return U64(x) * y;
#endif
}

u64 OVL auxMul(u32 x, u32 y, u64 carryIn, u32* carryOut) {
#if HAS_ASM
  u64 out;
  u32 co;
  // u32 dummy;
  __asm("v_mad_u64_u32 %[out], vcc, %[x], %[y], %[ci]\n\t"
        "v_addc_co_u32 %[co], vcc, 0, 0, vcc"
        : [out] "=v"(out), [co] "=v"(co)
        : [x] "v"(x), [y] "v"(y), [ci] "v"(carryIn) //, "{exec}"(dummy)
        : "vcc");
  *carryOut = co;
  return out;
#else
  u64 xy = U64(x) * y;
  u64 r = xy + carryIn;
  *carryOut = (r < carryIn);
  return r;
#endif
}

u128 wideMul(u64 x, u64 y) {
#if 1
  u64 p = auxMul(x, y);
  u64 q = auxMul(x, y >> 32, p >> 32);
  u32 co;  
  q = auxMul(x >> 32, y, q, &co);
  u64 r = auxMul(x >> 32, y >> 32, (U64(co) << 32) | (q >> 32));
  return U128(U64(p, q), r);
#else
  return U128(x) * y;
#endif
}

u64 reduce128(u128 x) { return add(U64(x), mul64w(x >> 64)); }

u64 mul(u64 a, u64 b) { return reduce128(wideMul(a, b)); }

u64 sq(u64 x) { return mul(x, x); }

u64 auxMulS(u32 x, u32 s, u32 ci) {
  return U64(x) * s + ci;
}

u128 wideMul3T4(u64 x) {
  // mul with 0xfffeffff'00000001 == (0xfffeffff<<32) + 1
  u32 a = x;
  u32 b = x >> 32;
  u64 c = auxMulS(a, 0xfffeffffu, b);
  u64 d = auxMulS(b, 0xfffeffffu, c >> 32);
  return (U128(d) << 64) | U64(a, c);
}

u128 wideMul7T8(u64 x) { return ((U128(x) << 32) - x) << 8; } // mul with 0xffffffff00

u64 mul1T4(u64 x) { return reduce128(U128(x) << 48); }
u64 mul3T4(u64 x) { return reduce128(wideMul3T4(x)); }

/*
u64 mul1T8(u64 x) { return reduce128(U128(x) << 24); }
u64 mul3T8(u64 x) { return reduce128(wideMul3T8(x)); }
u64 mul5T8(u64 x) { return reduce128(U128(x) * 0xfffffffeff000001); }
// { return reduce128(wideMul(x, 0xfffffffeff000001)); }
// 
u64 mul7T8(u64 x) { return reduce128(U128(x) * 0xfffffeff00000101); }
*/

u64 mul1T8(u64 x) { return reduce128(U128(x) * 0xfffffffeff000001); }
u64 mul3T8(u64 x) { return reduce128(U128(x) * 0xfffffeff00000101); }

u64 mul5T8(u64 x) { return reduce128(U128(x) << 24); }
u64 mul7T8(u64 x) { return reduce128(wideMul7T8(x)); }

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

u32 stepExtra(u32 extra) {
  u32 a = extra - (N - STEP);
  return I32(a) < 0 ? a + N : a;
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

Word doCarry64(i64 balanced, i64* outCarry, u32 extra) {
  u32 nBits = bitlenExtra(extra);
  assert(nBits < 32);
  Word w = lowBits(balanced, nBits);
  *outCarry = (balanced >> nBits) + (w < 0);
  return w;
}

OVL Word doCarry32(i64 balanced, i32* outCarry, u32 extra) {
  u32 nBits = bitlenExtra(extra);
  assert(nBits < 32);
  Word w = lowBits(balanced, nBits);
  assert((balanced >> (nBits + 32)) == 0 || (balanced >> (nBits + 32)) == -1); // verify that out-carry fits on 32 bits
  *outCarry = (balanced >> nBits) + (w < 0);
  return w;
}

OVL Word doCarry32(i32 balanced, i32* outCarry, u32 extra) {
  u32 nBits = bitlenExtra(extra);
  assert(nBits < 32);
  Word w = lowBits(balanced, nBits);
  assert((balanced >> (nBits + 32)) == 0 || (balanced >> (nBits + 32)) == -1); // verify that out-carry fits on 32 bits
  *outCarry = (balanced >> nBits) + (w < 0);
  return w;
}

Word carryStep(u64 u, i64* inOutCarry, u32 extra, u64 iWeight) {
  assert(u < PRIME);
  u = mul(u, iWeight);
  return doCarry64(balance(u) + *inOutCarry, inOutCarry, extra);
}

u64 carryWord(i64 w, i32* outCarry, u32 extra, u64 dWeight) {
  w = doCarry32(w, outCarry, extra);
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
#if 0
  X2(u[0], u[2]);
  X2(u[1], u[3]);
  u[3] = mul3T4(u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);
  SWAP(u[1], u[2]);
#else  
  SWAP(u[1], u[3]);
  dfft4(u);
#endif
}

void ifft4a(u64* u) {  
  dfft4(u);
  SWAP(u[1], u[3]);
}

void ifft8(u64* u) {
  X2(u[0], u[4]);
  X2(u[1], u[5]);
  X2(u[2], u[6]);
  X2(u[3], u[7]);
  u[5] = mul7T8(u[5]);
  u[6] = mul3T4(u[6]);
  u[7] = mul5T8(u[7]);

  X2(u[0], u[2]);
  X2(u[1], u[3]);
  u[3] = mul3T4(u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);

  X2(u[4], u[6]);
  X2(u[5], u[7]);
  u[7] = mul3T4(u[7]);
  X2(u[4], u[5]);
  X2(u[6], u[7]);

  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

void dfft8(u64* u) {
#if 1
  X2(u[0], u[4]);
  X2(u[1], u[5]);
  X2(u[2], u[6]);
  X2(u[3], u[7]);
  u[5] = mul1T8(u[5]);
  u[6] = mul1T4(u[6]);
  u[7] = mul3T8(u[7]);

  X2(u[0], u[2]);
  X2(u[1], u[3]);
  u[3] = mul1T4(u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);

  X2(u[4], u[6]);
  X2(u[5], u[7]);
  u[7] = mul1T4(u[7]);
  X2(u[4], u[5]);
  X2(u[6], u[7]);

  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
#else
  SWAP(u[1], u[7]);
  SWAP(u[2], u[6]);
  SWAP(u[3], u[5]);
  ifft8(u);
#endif
}

void shufl(u32 me, u32 WG, local u64* lds, u64* u, u32 n, u32 f) {
  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);
  
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i]; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i] = lds[i * WG + me]; }
}

typedef constant const u64* Trig;

void tabMulMem(u32 me, u32 WG, Trig trig, u64* u, u32 n, u32 f) {
  for (u32 i = 1; i < n; ++i) { u[i] = mul(u[i], trig[(me & ~(f-1)) + (i - 1) * WG]); }
}

void tabMulCpu(u32 me, u32 WG, Trig trig, u64* u, u32 n, u32 f) {
  u64 w = trig[(me & ~(f-1))];
  u64 step = w;
  for (u32 i = 1; i < n; ++i) { u[i] = mul(u[i], w); w = mul(w, step); }
}

void shuflAndMulMem(u32 me, u32 WG, local u64* lds, Trig trig, u64* u, u32 n, u32 f) {
  tabMulMem(me, WG, trig, u, n, f);
  shufl(me, WG, lds, u, n, f);
}

void shuflAndMulCpu(u32 me, u32 WG, local u64* lds, Trig trig, u64* u, u32 n, u32 f) {
  tabMulCpu(me, WG, trig, u, n, f);
  shufl(me, WG, lds, u, n, f);
}

// 256x4
void dFFT1K(u32 me, local u64* lds, u64* u, Trig trig) {
  // UNROLL_WIDTH_CONTROL
  for (i32 s = 0; s <= 6; s += 2) {
    if (s) { bar(); }
    dfft4(u);
    shuflAndMulMem(me, 256, lds, trig, u, 4, 1u << s);
  }
  dfft4(u);
}

void iFFT1K(u32 me, local u64* lds, u64* u, Trig trig) {
  // UNROLL_WIDTH_CONTROL
  for (i32 s = 0; s <= 6; s += 2) {
    if (s) { bar(); }
    ifft4(u);
    shuflAndMulMem(me, 256, lds, trig, u, 4, 1u << s);
  }
  ifft4(u);
}

#if NH == 4

// 1024x4
void dFFT4K(u32 me, local u64* lds, u64* u, Trig trig) {
  // UNROLL_WIDTH_CONTROL
  // __attribute__((opencl_unroll_hint(1)))
  for (i32 s = 0; s < 10; s += 2) {
    if (s) { bar(); }
    dfft4(u);
    shuflAndMulMem(me, 1024, lds, trig, u, NH, 1u << s);
  }
  dfft4(u);
}

void iFFT4K(u32 me, local u64* lds, u64* u, Trig trig) {
  // UNROLL_WIDTH_CONTROL
  // __attribute__((opencl_unroll_hint(1)))
  for (i32 s = 0; s < 10; s += 2) {
    if (s) { bar(); }
    ifft4(u);
    shuflAndMulMem(me, 1024, lds, trig, u, NH, 1u << s);
  }
  ifft4(u);
}

#elif NH == 8

// 512x8
void dFFT4K(u32 me, local u64* lds, u64* u, Trig trig) {
  for (i32 s = 0; s < 9; s += 3) {
    if (s) { bar(); }
    dfft8(u);
    shuflAndMulMem(me, 512, lds, trig, u, NH, 1u << s);
  }
  dfft8(u);
}

void iFFT4K(u32 me, local u64* lds, u64* u, Trig trig) {
  for (i32 s = 0; s < 9; s += 3) {
    if (s) { bar(); }
    ifft8(u);
    shuflAndMulMem(me, 512, lds, trig, u, NH, 1u << s);
  }
  ifft8(u);
}

#else
#error NH
#endif

#define ATTR(x) __attribute__((x))
#define WGSIZE(n) ATTR(reqd_work_group_size(n, 1, 1))

#define P(x) global x * restrict
#define CP(x) const P(x)

kernel WGSIZE(WIDTH) void sinkCarry(P(i32) out, CP(i32) in, CP(i64) inCarry) {
  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 gm1 = (HEIGHT / 4 - 1 + gr) % (HEIGHT / 4);
  u32 mm1 = (WIDTH - 1 + me) % WIDTH;
  i64 carry64 = inCarry[WIDTH * gm1 + (gr ? me : mm1)];
  i32 carry = 0;
  u32 extra = extraK(4 * gr + HEIGHT * me);
  u32 pos = 4 * WIDTH * gr + me;
  out[pos]  = doCarry32(in[pos] + carry64, &carry, extra);
  for (int i = 1; i < 3; ++i) {
    extra = stepExtra(extra);
    pos = 4 * WIDTH * gr + WIDTH * i + me;
    out[pos] = doCarry32(in[pos] + carry, &carry, extra);
  }
  pos = 4 * WIDTH * gr + WIDTH * 3 + me;
  out[pos] = in[pos] + carry;  
}

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


kernel WGSIZE(G_H) void tailSquare(P(u64) io, Trig dSmallTrig, Trig iSmallTrig) {
  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  local u64 lds[HEIGHT];

  u64 u[NH];

  for (int i = 0; i < NH; ++i) { u[i] = io[HEIGHT * gr + G_H * i + me]; }

  dFFT4K(me, lds, u, dSmallTrig);

  bar();
  
  for (int i = 0; i < NH; ++i) { u[i] = sq(u[i]); }
  
  iFFT4K(me, lds, u, iSmallTrig);

  for (int i = 0; i < NH; ++i) { io[HEIGHT * gr + G_H * i + me] = u[i]; }
}

kernel WGSIZE(HEIGHT/NH) void tailMul(P(u64) io, CP(u64) inB, Trig dSmallTrig, Trig iSmallTrig) {
  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  local u64 lds[HEIGHT];

  u64 u[NH], v[NH];

  for (int i = 0; i < NH; ++i) { u[i] =  io[HEIGHT * gr + G_H * i + me]; }
  for (int i = 0; i < NH; ++i) { v[i] = inB[HEIGHT * gr + G_H * i + me]; }

  dFFT4K(me, lds, u, dSmallTrig);

  bar();

  dFFT4K(me, lds, v, dSmallTrig);

  bar();
  
  for (int i = 0; i < NH; ++i) { u[i] = mul(u[i], v[i]); }

  iFFT4K(me, lds, u, iSmallTrig);

  for (int i = 0; i < NH; ++i) { io[HEIGHT * gr + G_H * i + me] = u[i]; }
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

/*
kernel WGSIZE(1024) void transposeCarryOut(P(i64) out, P(i64) in) {
  local i64 lds[1024];
  trans64(out, in, WIDTH, HEIGHT / 4, lds);
}
*/

// Generate a small unused kernel so developers can look at how well individual macros assemble and optimize
kernel WGSIZE(64) void testKernel(global u64* io) {
  uint me = get_local_id(0);
  u64 a[NH], b[NH];
  if (me == 0) {
    for (int i = 0; i < NH; ++i) { a[i] = b[i] = io[i]; }

    ifft4(a);
    ifft4a(b);
    
    for (int i = 0; i < NH; ++i) {
      io[i] = a[i];
      io[i + NH] = b[i];
    }
  }
}

// Copyright (C) 2017-2018 Mihai Preda.

#pragma OPENCL FP_CONTRACT ON

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

#define assert(cond, value)
// if (!(cond) /*&& (get_local_id(0) == 0)*/) { printf("assert #%d: %x\n", __LINE__, (uint) value); }
// #define printf DONT_USE

// #define B13 (1ul | (1ul << 13) | (1ul << 26) | (1ul << 39) | (1ul << 52))
#define B17 (1ul | (1ul << 17) | (1ul << 34) | (1ul << 51))
#define B19 (1ul | (1ul << 19) | (1ul << 38) | (1ul << 57))
#define B23 (1ul | (1ul << 23) | (1ul << 46))
#define B29 (1ul | (1ul << 29) | (1ul << 58))
#define B31 (1ul | (1ul << 31) | (1ul << 62))

uint rem(int x, uint p, int inv) {
  int a = x - mul_hi(x, inv) * p;
  uint r = (a < 0) ? a + p : a;
  assert(r < p, r);
  return r;
}

uint modStep(int bit, int p, int step) {
  assert(p > 0, p);
  int a = bit - step;
  uint r = (a < 0) ? a + p : a;
  assert(r < p, r);
  return r;
}

void bar()    { barrier(CLK_LOCAL_MEM_FENCE); }

ulong shl64(ulong x, uint n) { return (n < 64) ? x << n : 0; }
uint shl32(uint x, uint n) { return (n < 32) ? x << n : 0; }

#define SIEVE2(i, mask) filter(tab[i*3], words, rem(btcs[i] - threadStart, tab[i*3], tab[i*3+1]), mask, tab[i*3+2])
#define SIEVE1(i) filter(tab[i*3], words, rem(btcs[i] - threadStart, tab[i*3], tab[i*3+1]), 1 | (1ul << tab[i*3]), tab[i*3+2])
#define SIEVE0(i) filter(tab[i*3], words, rem(btcs[i] - threadStart, tab[i*3], tab[i*3+1]), 1, tab[i*3+2])

#define SIEVE_WG 1024
#define LDS_BITS (LDS_WORDS * 32)
#define THREAD_WORDS (LDS_WORDS / SIEVE_WG)
#define THREAD_DWORDS (THREAD_WORDS / 2)

void filter(uint prime, ulong *words, uint pos, ulong mask, int step) {
  for (int i = 0; i < THREAD_DWORDS; ++i) {
    words[i] |= shl64(mask, pos);
    pos = modStep(pos, prime, step);
  }
}

#define P(x) x, 0xffffffffu / x, 64 * SIEVE_WG % x

KERNEL(SIEVE_WG) sieve(const global uint * const primes, const global uint * const invs,
                 const global int * const btcs, global uint *outN, global uint *outK) {
  uint g = get_group_id(0);
  uint me = get_local_id(0);

  local uint lds[LDS_WORDS];
  
  uint threadStart = LDS_BITS * g + 64 * me;

  {
    ulong words[THREAD_DWORDS] = {0};
    
    uint tab[SPECIAL_PRIMES * 3] = {
// P( 13),
        P( 17), P( 19), P( 23), P( 29), P( 31), P( 37), P( 41), P( 43), P( 47), P( 53), P( 59), P( 61), P( 67), P( 71), P( 73), 
P( 79), P( 83), P( 89), P( 97), P(101), P(103), P(107), P(109), P(113), P(127), P(131), P(137), P(139), P(149), P(151), P(157),
P(163),
//         P(167), P(173), P(179), P(181), P(191), P(193), P(197), P(199), P(211), P(223), P(227), P(229), P(233), P(239), P(241), 
// P(251), P(257), P(263), P(269), P(271), P(277), P(281), P(283), P(293), P(307), P(311), P(313), P(317), P(331), P(337), P(347), 
// P(349), P(353), P(359), P(367), P(373), P(379), P(383), P(389), P(397), P(401), P(409), P(419), P(421), P(431), P(433), P(439), 
// P(443), P(449), P(457), P(461), P(463), P(467), P(479), P(487), P(491), P(499), P(503), P(509), P(521), P(523), P(541), P(547), 
// P(557), P(563), P(569), P(571), P(577), P(587), P(593), P(599), P(601), P(607), P(613), P(617), P(619), P(631), P(641), P(643), 
// P(647), P(653), P(659), P(661), P(673), P(677), P(683), P(691), P(701), P(709), P(719), P(727), P(733), P(739), P(743), P(751),
    };
    
    // SIEVE2(0, B13);
    SIEVE2(0, B17);
    SIEVE2(1, B19);
    SIEVE2(2, B23);
    SIEVE2(3, B29);
    SIEVE2(4, B31);
    for (int i =  5; i < min(12u, SPECIAL_PRIMES); ++i) { SIEVE1(i); }
    for (int i = 12; i < SPECIAL_PRIMES; ++i) { SIEVE0(i); }

    local ulong *bigLds = (local ulong *)lds;
    for (int i = 0; i < THREAD_DWORDS; ++i) {
      bigLds[SIEVE_WG * i + me] = words[i];
    }
  }

  bar();

  for (int i = 0; i < (NPRIMES - SPECIAL_PRIMES) / SIEVE_WG; ++i) {
    uint pos = SPECIAL_PRIMES + SIEVE_WG * i + me;
    uint prime = primes[pos];
    uint inv   = invs[pos];
    int btc    = btcs[pos];
    for (uint pos = rem(btc - LDS_BITS * g, prime, inv); pos < LDS_BITS; pos += prime) {
      atomic_or(&lds[pos / 32], 1 << (pos % 32));
    }
  }
  
  bar();

  uint count = 0;
  
  for (int i = 0; i < THREAD_WORDS; ++i) {
    count += popcount(~lds[SIEVE_WG * i + me]);
  }

  uint save0, save1;
  
  if (me == 0) {
    save0 = lds[0];
    save1 = lds[1];
    lds[0] = (uint) -1;
    lds[1] = 0;
  }

  bar();

  atomic_min(&lds[0], count);

  uint raw = atomic_add(&lds[1], count | (1 << 20));
  
  bar();

  uint common = lds[0];
  // uint ord = raw >> 20;
  uint mePos = (raw & 0xfffff) - (raw >> 20) * common;

  bar();

  if (me == 0) {
    uint groupCount = lds[1] & 0xfffff;
    lds[0] = atomic_add(outN, groupCount);
  }

  bar();

  uint groupPos = lds[0];
  outK += groupPos;
  
  if (me == 0) { lds[1] = save1; }
  
  bar();
  
  if (me == 0) { lds[0] = save0; }

  int wordPos = 0;
  uint word = ~lds[me];
  for (int i = 0; i < common; ++i) {
    while (word == 0) { word = ~lds[me + SIEVE_WG * ++wordPos]; }
    uint bit = ctz(word);
    word &= word - 1; // clear last bit set.
    outK[i * SIEVE_WG + me] = (LDS_WORDS * g + SIEVE_WG * wordPos + me) * 32 + bit;
  }

  for (int i = 0; i < count - common; ++i) {
    while (word == 0) { word = ~lds[me + SIEVE_WG * ++wordPos]; }
    uint bit = ctz(word);
    word &= word - 1; // clear last bit set.
    outK[SIEVE_WG * common + i + mePos] = (LDS_WORDS * g + SIEVE_WG * wordPos + me) * 32 + bit;    
  }
}

#define OVER __attribute__((overloadable))

ulong mad64(uint a, uint b, ulong c) { return a * (ulong) b + c; }

typedef unsigned long long u128;

u128 OVER to128(uint3 u) { return (((u128) u.z) << 64) | (((ulong) u.y) << 32) | u.x; }
u128 OVER to128(uint4 u) { return (((u128) u.w) << 96) | to128(u.xyz); }
uint3 toUint3(u128 a) { return (uint3) (a, a >> 32, a >> 64); }
uint4 toUint4(u128 a) { return (uint4) (toUint3(a), a >> 96); }
ulong toLong(uint2 a) { return (((ulong) a.y) << 32) | a.x; }
uint2 toUint2(long a) { return (uint2)(a, a >> 32); }

uint2 OVER add(uint2 a, ulong delta) { return toUint2(toLong(a) + delta); }
uint3 OVER add(uint3 a, uint3 b) { return toUint3(to128(a) + to128(b)); }
uint3 OVER add(uint3 a, uint x) { return toUint3(to128(a) + x); }
uint4 OVER sub(uint4 a, uint4 b) { return toUint4(to128(a) - to128(b)); }
uint3 OVER sub(uint3 a, uint3 b) { return toUint3(to128(a) - to128(b)); }
uint4 OVER mul(uint3 n, uint q) { return toUint4(to128(n) * q); }
bool equal(uint3 a, uint3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

uint alignbit(uint2 a, uint k) {
  assert(k <= 31, k);
  return toLong(a) >> (k & 31); // Squeeze a v_alignbit_b32 from the compiler.
}

uint3 OVER lshift(uint3 u, uint k) {
  assert(k > 0 && k <= 32, k);
  k = (32 - k) & 31;
  return (uint3) (alignbit((uint2)(0, u.x), k), alignbit(u.xy, k), alignbit(u.yz, k));
}

uint4 OVER lshift(uint4 u, uint k) {
  assert(k > 0 && k <= 32, k);
  k = (32 - k) & 31;
  return (uint4) (alignbit((uint2)(0, u.x), k), alignbit(u.xy, k), alignbit(u.yz, k), alignbit(u.zw, k));
}

uint3 OVER rshift(uint4 u, uint k) {
  assert(k <= 31, k);
  k = k & 31; // let the compiler know the range.
  return (uint3) (alignbit(u.xy, k), alignbit(u.yz, k), alignbit(u.zw, k));
}

float OVER toFloat(uint x, float y) { return y * (1l << 32) + x; }
float OVER toFloat(uint2 u) { return toFloat(u.x, u.y); }
float OVER toFloat(uint3 u) { return toFloat(u.x, toFloat(u.y, u.z)); }
float floatInv(uint2 u) { return as_float(as_uint(1 / toFloat(u)) - 4); } // lower bound.

// See mfaktc: tf_barrett96_div.cu
uint3 div192(uint3 n) {
  assert(n.z != 0, n.z);

  uint3 res;
  
  float nf = floatInv(n.yz);
  // if (get_local_id(0)==0) { printf("%x %x %g\n", n.y, n.z, nf); }

  uint8 q = (uint8) (0, 0, 0, 0, 0, 1 << (31 - clz(n.z)), 0, 0);
  
  // step 1
  float qf = toFloat(0, q.s5) * (1 << 21); // float qf = ldexp(toFloat(q.s4, q.s5), 21);
  uint  qi = qf * nf;
  assert(qi < (1<<21), qi);
  
  qi <<= 11;
  res.z = qi;

  uint4 nn = mul(n, qi);
  q.s2345 = sub(q.s2345, nn);

  // if (get_local_id(0)==0) { printf("1: %x %x %x %x\n", q.s2, q.s3, q.s4, q.s5); }
  
  // step 2
  qf = toFloat(q.s345) * (1 << 9);
  qi = qf * nf;
  // if (get_local_id(0)==0) { printf("%d %g %g\n", qi, qf, nf); }
  assert(qi < (1<<22), qi);
  res.y = qi << 23;
  res.z += qi >> 9;

  // assert(q.s5 == 
  
  nn = lshift(mul(n, qi), 23);
  q.s1234 = sub(q.s1234, nn);

  /*
  if (get_local_id(0)==0) {
    printf("%x  nn: %x %x %x %x\n", qi, nn.x, nn.y, nn.z, nn.w);
    printf("2: %x %x %x %x\n", q.s1, q.s2, q.s3, q.s4);
  }
  */
  
  // step 3
  qf = toFloat(q.s34) * (1 << 29);
  qi = qf * nf;

  // if (qi >= (1<<22) && get_local_id(0) == 0) { printf("n %x%08x%08x %g %g %u %x %x\n", n.z, n.y, n.x, nf, qf, qi, q.s4, q.s3); }
  
  assert(qi < (1<<22), qi);
  qi <<= 3;  
  res.yz = add(res.yz, qi);

  nn = mul(n, qi);
  q.s1234 = sub(q.s1234, nn);

  // step 4
  qf = toFloat(q.s234) * (1 << 17);
  qi = qf * nf;
  assert(qi < (1<<22), qi);
  res.x = qi << 15;
  res.yz = add(res.yz, qi >> 17);

  nn = mul(n, qi);
  nn = lshift(nn, 15);
  q.s0123 = sub(q.s0123, nn);
  
  // step 5
  qf = toFloat(q.s123);
  qi = qf * nf;
  assert(qi < (1<<20), qi);
  res.xyz = add(res.xyz, (uint3)(qi, 0, 0));
  
  return res;
}

uint8 square(uint3 u) {
  uint c = 2 * u.z;
  ulong ab = u.x * (ulong) u.y;
  ulong abLow = ab << 33;
  ulong aa = mad64(u.x, u.x, abLow);
  ulong abHigh = (ab >> 31) + (aa < abLow);
  ulong ac = mad64(u.x, c, abHigh);
  ulong bb = mad64(u.y, u.y, ac);
  ulong bc = mad64(u.y, c, bb >> 32);
  uint bcHigh = ((uint) (bc >> 32)) + (bb < ac);
  ulong cc = mad64(u.z, u.z, bcHigh);
  return (uint8)(aa, aa >> 32, bb, bc, cc, cc >> 32, 0, 0);
}

uint3 mulLow(uint3 u, uint3 v) { return toUint3(to128(u) * to128(v)); }

uint3 mulHi(uint3 u, uint3 v) {
  return toUint3(mul_hi(u.x, v.z) + (ulong) mul_hi(u.z, v.x) + ((toLong(u.yz) * (u128) toLong(v.yz)) >> 32));
}

// b % m, given u the "inverse" of m.
uint3 mod(uint8 b, uint3 m, uint3 u) {
  return sub(b.xyz, mulLow(m, mulHi(rshift(b.s2345, (31 - clz(m.z))), u)));
}

bool isFactor(uint flushed, uint3 m, uint4 preshifted) {
  uint8 b = (uint8) (0, 0, preshifted, 0, 0);  
  uint3 u = div192(m);
  uint3 a = mod(b, m, u);

  do {
    a = mod(square(a), m, u);
    if (flushed >> 31) { a = lshift(a, 1); }
  } while (flushed <<= 1);
  
  uint n = toFloat(a.yz) * floatInv(m.yz);
  a = sub(a, mul(m, n).xyz);
  return equal(a, (uint3)(1, 0, 0)) || equal(a, add(m, 1));
}

#define TF_WG 1024
KERNEL(TF_WG) tf(int N, uint exponent, ulong kBase, global uint *bufK, global ulong *bufFound) {
  assert(exponent & 1, exponent);
  uint flushed = exponent << clz(exponent); // flush left.
  uint topBits = ((flushed >> 24) < 178) ? 8 : 7;
  uint shift   = flushed >> (32 - topBits);
  assert(shift >= 89 && shift < 178, shift);
  flushed <<= topBits;
  uint4 preshifted = (uint4) (toUint2(shl64(1, shift - 64)), toUint2(shl64(1, shift - 128)));
  
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    uint kBit = bufK[i];
    ulong k = kBase + kBit * (ulong) NCLASS;    
    uint3 m = toUint3(2 * exponent * (u128) k + 1);
    if (isFactor(flushed, m, preshifted)) { bufFound[0] = k; }
  }
}

KERNEL(1024) initBtc(uint N, uint exp, ulong k, global uint *primes, global uint *invs, global uint *outBtc) {
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    uint prime = primes[i];
    uint inv   = invs[i];
    uint qMod  = (2 * exp * (k % prime) + 1) % prime;
    uint btc   = (prime - qMod) * (ulong) inv % prime;
    assert(btc < prime, btc);
    // assert(2 * exp * (u128) (k + ((ulong) btc) * NCLASS) % prime == prime - 1, 0);
    outBtc[i] = btc;
  }
}

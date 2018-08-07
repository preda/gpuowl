// Copyright (C) 2017-2018 Mihai Preda.

#pragma OPENCL FP_CONTRACT ON

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

#define assert(cond, value)
// if (!(cond) && (get_local_id(0) == 0)) { printf("assert #%d: %x\n", __LINE__, (uint) value); }

#define printf error

// #define P(format, ...) if (get_local_id(0)==0) { printf(format, ##__VA_ARGS__); }

#define B13 (1ul | (1ul << 13) | (1ul << 26) | (1ul << 39) | (1ul << 52))
#define B17 (1ul | (1ul << 17) | (1ul << 34) | (1ul << 51))
#define B19 (1ul | (1ul << 19) | (1ul << 38) | (1ul << 57))
#define B23 (1ul | (1ul << 23) | (1ul << 46))
#define B29 (1ul | (1ul << 29) | (1ul << 58))
#define B31 (1ul | (1ul << 31) | (1ul << 62))

int rem(int x, uint p, int inv) {
  int a = x - mul_hi(x, inv) * p;
  return (a < 0) ? a + p : a;
}

int modP(int x, uint p) { return rem(x, p, 0xffffffff / p); }

  /*
  int a = x - mul_hi(x, (int) (0xffffffff / p + 1)) * p;
  int b = a - p;
  return (b >= 0) ? b : a;
  */

uint modStep(int bit, int delta, int p) {
  int a = bit - delta % p;
  return (a < 0) ? a + p : a;
}

void bar()    { barrier(CLK_LOCAL_MEM_FENCE); }

#define WG 512

ulong shl64(ulong x, uint n) { return (n < 64) ? x << n : 0; }
uint shl32(uint x, uint n) { return (n < 32) ? x << n : 0; }

#define SIEVE2(i, p) filter(p, words, rem(btcs[i] - threadStart, p, invs[i]), 1 | shl64(1, p) | shl64(1, 2 * p) | shl64(1, 3 * p))
#define SIEVE1(i, p) filter(p, words, rem(btcs[i] - threadStart, p, invs[i]), 1 | (1ul << p))
#define SIEVE0(i, p) filter(p, words, rem(btcs[i] - threadStart, p, invs[i]), 1)

#define LDS_WORDS (8 * 1024)
#define LDS_BITS (LDS_WORDS * 32)
#define THREAD_WORDS (LDS_WORDS / WG)
#define THREAD_DWORDS (THREAD_WORDS / 2)

void filter(uint prime, ulong *words, uint pos, ulong mask) {
  for (int i = 0; i < THREAD_DWORDS; ++i) {
    words[i] |= shl64(mask, pos);
    pos = modStep(pos, 64 * WG, prime);
  }
}

KERNEL(WG) sieve(const global uint * const primes, const global uint * const invs,
                 const global int * const btcs, global uint *outN, global uint *outK) {
  uint g = get_group_id(0);
  uint me = get_local_id(0);

  local uint lds[LDS_WORDS];
  local ulong *bigLds = (local ulong *)lds;
  
  uint threadStart = LDS_BITS * g + 64 * me;
  // uint threadStart = LDS_BITS * g + (LDS_BITS / WG) * me;

  {
    ulong words[THREAD_DWORDS];

    uint pos = modP(btcs[0] - threadStart, 13);
    for (int i = 0; i < THREAD_DWORDS; ++i) {
      words[i] = B13 << pos;
      pos = modStep(pos, 64 * WG, 13);
    }


#define SWITCH 64
    
    uint primes[] = {
      13,   17,  19,  23,  29,  31,  37,  41,
      43,   47,  53,  59,  61,  67,  71,  73,
      79,   83,  89,  97, 101, 103, 107, 109,
      113, 127, 131, 137, 139, 149, 151, 157,
      163, 167, 173, 179, 181, 191, 193, 197,
      199, 211, 223, 227, 229, 233, 239, 241,
      251, 257, 263, 269, 271, 277, 281, 283,
      293, 307, 311, 313, 317, 331, 337, 347,
    };

    // for (int i = 1; i < 6; ++i) { SIEVE2(i, primes[i]); }
    SIEVE2(1, primes[1]);
    SIEVE2(2, primes[2]);
    SIEVE2(3, primes[3]);
    SIEVE2(4, primes[4]);
    SIEVE2(5, primes[5]);
    for (int i =  6; i < 13; ++i) { SIEVE1(i, primes[i]); }
    for (int i = 13; i < SWITCH; ++i) { SIEVE0(i, primes[i]); }

    for (int i = 0; i < THREAD_DWORDS; ++i) {
      // bigLds[me * 16 + i] = words[i];
      bigLds[WG * i + me] = words[i];
    }
  }

  bar();

  for (int i = 0; i < 45; ++i) {
    uint prime = primes[SWITCH + WG * i + me];
    uint inv = invs[SWITCH + WG * i + me];
    int btc = btcs[SWITCH + WG * i + me];    
    for (uint pos = rem(btc - LDS_BITS * g, prime, inv); pos < LDS_BITS; pos += prime) {
      atomic_or(&lds[pos / 32], 1 << (pos % 32));
      
      // if ((lds[pos / 32] & mask) == 0) { lds[pos / 32] |= mask; }
      // uint read = lds[pos / 32];
      // if ((read & mask) == 0) { lds[pos / 32] = read | mask; }
      // lds[pos / 32] |= mask;
    }
  }

  for (int i = 45; i < 2 * 1024; ++i) {
    uint prime = primes[SWITCH + WG * i + me];
    uint inv = invs[SWITCH + WG * i + me];
    int btc = btcs[SWITCH + WG * i + me];    
    uint pos = rem(btc - LDS_BITS * g, prime, inv);
    if (pos < LDS_BITS) { atomic_or(&lds[pos / 32], 1 << (pos % 32)); }
  }
  
  bar();

  uint count = 0;
  
  for (int i = 0; i < THREAD_WORDS; ++i) {
    count += popcount(~lds[WG * i + me]);
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

#if 1
  int wordPos = 0;
  uint word = ~lds[me];
  for (int i = 0; i < common; ++i) {
    while (word == 0) { word = ~lds[me + WG * ++wordPos]; }
    uint bit = ctz(word);
    word &= word - 1; // clear last bit set.
    outK[i * WG + me] = (LDS_WORDS * g + WG * wordPos + me) * 32 + bit;
  }

  for (int i = 0; i < count - common; ++i) {
    while (word == 0) { word = ~lds[me + WG * ++wordPos]; }
    uint bit = ctz(word);
    word &= word - 1; // clear last bit set.
    outK[WG * common + i + mePos] = (LDS_WORDS * g + WG * wordPos + me) * 32 + bit;    
  }
#endif
  
  /*
  uint pos = groupPos + mePos + 1;
  for (int i = 0; i < 32; ++i) {
    uint w = lds[WG * i + me];
    for (int b = 0; b < 32; ++b) {
      if ((w & (1u << b)) == 0) {
        out[pos++] = (g * 8 * 1024 + i * WG + me) * 32 + b;
      }
    }
  }
  */
}

#define OVER __attribute__((overloadable))

// struct uint6 { uint x, y, z, w, d4, d5; };

ulong mad64(uint a, uint b, ulong c) { return a * (ulong) b + c; }
/*
    ulong r;
  __asm("v_mad_u64_u32 %0, vcc, %1, %2, %3\n"
        : "=v"(r)
        : "v"(a), "v"(b), "v"(c)
        : "vcc");
  return r;
*/

ulong mul64(uint a, uint b) { return mad64(a, b, 0); }
/*
  ulong r;
  __asm("v_mad_u64_u32 %0, vcc, %1, %2, 0\n"
        : "=v"(r)
        : "v"(a), "v"(b)
        : "vcc");
  return r;
*/

uint4 OVER mul(uint3 n, uint q) {
  // return toUint4(toU128(n) * q);
  
  ulong a = mul64(n.x, q);
  ulong b = mad64(n.y, q, a >> 32);
  ulong c = mad64(n.z, q, b >> 32);
  return (uint4)(a, b, c, c >> 32);
}

#define USE128 1

#if USE128

typedef unsigned long long u128;
u128 OVER to128(uint3 u) { return (((u128) u.z) << 64) | (((ulong) u.y) << 32) | u.x; }
u128 OVER to128(uint4 u) { return (((u128) u.w) << 96) | to128(u.xyz); }
uint3 toUint3(u128 a) { return (uint3) (a, a >> 32, a >> 64); }
uint4 toUint4(u128 a) { return (uint4) (toUint3(a), a >> 96); }

#endif

#if 0
uint3 add(uint3 a, uint3 b) {
  ulong x = a.x + (ulong) b.x;
  ulong y = a.y + (ulong) b.y + (x >> 32);
  uint  z = a.z + b.z + (y >> 32);
  return (uint3) (x, y, z);
}
#endif

ulong toLong(uint2 a) { return (((ulong) a.y) << 32) | a.x; }
uint2 toUint2(long a) { return (uint2)(a, a >> 32); }

uint2 OVER add(uint2 a, ulong delta) { return toUint2(toLong(a) + delta); }
uint3 OVER add(uint3 a, uint3 b) { return toUint3(to128(a) + to128(b)); }
uint3 OVER add(uint3 a, uint x) { return toUint3(to128(a) + x); }
uint4 OVER sub(uint4 a, uint4 b) { return toUint4(to128(a) - to128(b)); }
uint3 OVER sub(uint3 a, uint3 b) { return toUint3(to128(a) - to128(b)); }
bool equal(uint3 a, uint3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

uint alignbit(uint2 a, uint k) {
  assert(k <= 31, k);
  return toLong(a) >> (k & 31); // Squeeze a v_alignbit_b32 from the compiler.
}
// uint align(uint2 a, uint k) { return toLong(a) >> (32 - k); }

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


// float OVER toFloat(ulong x) { return ((float) (x >> 32)) * (1l << 32) + (uint) x; }
float OVER toFloat(uint x, float y) { return y * (1l << 32) + x; }
float OVER toFloat(uint2 u) { return toFloat(u.x, u.y); }
float OVER toFloat(uint3 u) { return toFloat(u.x, toFloat(u.y, u.z)); }
float floatInv(uint2 u) { return as_float(as_uint(1 / toFloat(u)) - 3); } // lower bound.

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

uint8 square94(uint3 u) {
  uint c = 2 * u.z;
  ulong ab = mad64(u.x, u.y, 0);
  ulong abLow = ab << 33;
  ulong aa = mad64(u.x, u.x, abLow);
  ulong abHigh = (ab >> 31) + (aa < abLow);
  ulong ac = mad64(u.x, c, abHigh);
  ulong bb = mad64(u.y, u.y, ac);
  ulong bc = mad64(u.y, c, bb >> 32);
  uint bcHigh = ((uint) (bc >> 32)) + (bb < ac);
  ulong cc = mad64(u.z, u.z, bcHigh);
  return (uint8)((uint) aa, aa >> 32, (uint) bb, (uint) bc, (uint) cc, cc >> 32, 0, 0);
}

uint3 mulLow(uint3 u, uint3 v) {
#if USE128
  return toUint3(to128(u) * to128(v));
#else
  ulong ad = mul64(u.x, v.x);
  ulong ae = mad64(u.x, v.y, ad >> 32);
  ulong db = mad64(u.y, v.x, ae);
  uint z = u.y * v.y + u.x * v.z + u.z * v.x + (uint) (db >> 32);
  return (uint3)(ad >> 32, db >> 32, z);
#endif
}

uint3 mulHi(uint3 u, uint3 v) {
  ulong t = mul_hi(u.x, v.z) + (ulong) mul_hi(u.z, v.x);
  
#if USE128
  return toUint3(((toLong(u.yz) * (u128) toLong(v.yz)) >> 32) + t);
#else
  ulong a  = mul_hi(u.y, v.y) + t;
  ulong bf = mad64(u.y, v.z, a);
  ulong ec = mad64(u.z, v.y, bf);
  ulong cf = mad64(u.z, v.z, (ec >> 32) + (bf < a) + (ec < bf));
  return (uint3) (ec, cf, cf >> 32);
#endif
}

// b % m, given u the "inverse" of m.
uint3 mod(uint8 b, uint3 m, uint3 u) {
  return sub(b.xyz, mulLow(m, mulHi(rshift(b.s2345, (31 - clz(m.z))), u)));
}

bool isFactor(uint exp, uint3 m) {
  // int shift = 31 - clz(m.z);
  exp <<= clz(exp); // flush left.
  uint topBits = (((exp << 1) >> 30)) ? 7 : 8;
  uint preShift = exp >> (32 - topBits);
  assert(preShift >= 80 && preShift < 160, preShift);
  exp <<= topBits;

  // if (get_local_id(0) == 0) { printf("preShift %d\n", preShift); }
  
  uint8 b = (uint8) (0, 0, toUint2(shl64(1, preShift - 64)), shl32(1, preShift - 128), 0, 0, 0);
  
  uint3 u = div192(m);
  uint3 a = mod(b, m, u);

  do {
    a = mod(square94(a), m, u);
    if (exp >> 31) { a = lshift(a, 1); }
    // if (get_local_id(0) == 0) { printf("%x%08x%08x\n", a.z, a.y, a.x); }
  } while (exp <<= 1);
  uint n = toFloat(a.yz) * floatInv(m.yz);
  // if (get_local_id(0) == 0) { printf("%.20g %.20g %u\n", toFloat(a.yz), floatInv(m.yz), n); }
  a = sub(a, mul(m, n).xyz);
  return equal(a, (uint3)(1, 0, 0)) || equal(a, add(m, 1));
}

#define NCLASS 4620

KERNEL(1024) tf(int N, uint exp, ulong kBase, global uint *bufK, global ulong *bufFound) {
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    uint kBit = bufK[i];
    ulong k = kBase + kBit * (ulong) NCLASS;    
    uint3 m = toUint3(2 * exp * (u128) k + 1);
    if (isFactor(exp, m)) { bufFound[0] = k; }
  }
}

KERNEL(1024) initBtc(uint N, uint exp, ulong k, global uint *primes, global uint *invs, global uint *outBtc) {
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    uint prime = primes[i];
    uint inv   = invs[i];
    uint qMod  = (2 * exp * (k % prime) + 1) % prime;
    uint btc   = (prime - qMod) * (ulong) inv % prime;
    assert(btc < prime, btc);
    // assert(2 * exp * u128(k + u64(btc) * NCLASS) % prime == prime - 1);
    outBtc[i] = btc;
  }
}

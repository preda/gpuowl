// Prime M(31) == 2^31 - 1
#define M31 0x7fffffffu

uint lo(ulong a) { return a & 0xffffffff; }
uint up(ulong a) { return a >> 32; }

// input 32 bits except 2^32-1; output 31 bits.
uint mod(uint x) { return (x >> 31) + (x & M31); }

// input 63 bits; output 31 bits.
uint bigmod(ulong x) {
  x = u64(2 * up(x)) + lo(x); // 'up' is 31 bits.
  return mod(2 * up(x) + lo(x));
}

// negative: -x; input 31 bits.
uint neg(uint x) { return M31 - x; }
  
uint add1(uint a, uint b) { return mod(a + b); }
uint sub1(uint a, uint b) { return add1(a, neg(b)); }

T2 add(T2 a, T2 b) { return U2(add1(a.x, b.x), add1(a.y, b.y)); }
T2 sub(T2 a, T2 b) { return U2(sub1(a.x, b.x), sub1(a.y, b.y)); }

// Input can be full 32bits. k <= 30 (i.e. mod 31).
uint shl1(uint a,  uint k) { return bigmod(u64(a) << k); }

ulong wideMul(uint a, uint b) { return u64(a) * b; }

// The main, complex multiplication; input and output 31 bits.
// (a + i*b) * (c + i*d) mod reduced.
T2 mul(T2 u, T2 v) {
  uint a = u.x, b = u.y, c = v.x, d = v.y;
  ulong k1 = wideMul(c, add1(a, b));
  ulong k2 = wideMul(a, sub1(d, c));
  ulong k3 = wideMul(b, neg(add1(d, c)));
  // k1..k3 have at most 62 bits, so sums are at most 63 bits.
  return U2(bigmod(k1 + k3), bigmod(k1 + k2));
}

// scalar mul.
uint mul1(uint a, uint b) { return bigmod(wideMul(a, b)); }

// input, output 31 bits. Uses (a + i*b)^2 == ((a+b)*(a-b) + i*2*a*b).
T2 sq(T2 a) { return U2(mul1(a.x + a.y, sub1(a.x, a.y)), mul1(a.x, a.y << 1)); }

// Prime M(31) == 2^31 - 1
#define M31 0x7fffffffu

// uint lo(ulong a) { return a & 0xffffffffu; }
uint up(ulong a) { return a >> 32; }

// input 32 bits except 2^32-1; output 31 bits.
uint mod(uint x) { return (x >> 31) + (x & M31); }

// input 62 bits.
uint mod62(ulong x) {
  uint a = up(x << 1); // 31 bits
  uint b = x & M31;    // 31 bits
  return mod(a + b);
}

// negative (-x). input 31 bits.
uint neg(uint x) { return M31 - x; }
  
uint add1(uint a, uint b) { return mod(a + b); }
uint sub1(uint a, uint b) { return add1(a, neg(b)); }

uint2 add(uint2 a, uint2 b) { return U2(add1(a.x, b.x), add1(a.y, b.y)); }
uint2 sub(uint2 a, uint2 b) { return U2(sub1(a.x, b.x), sub1(a.y, b.y)); }

// k <= 30 (i.e. mod 31). Assumes bits(a) + k <= 62.
uint shl1(uint a,  uint k) { return mod62(u64(a) << k); }

// if both inputs are 31 bits, output is 62 bits.
ulong wideMul(uint a, uint b) { return u64(a) * b; }

uint mul1(uint a, uint b) { return mod62(wideMul(a, b)); }

// The main, complex multiplication; input and output 31 bits.
// (a + i*b) * (c + i*d) mod reduced.
uint2 mul(uint2 u, uint2 v) {
  uint a = u.x, b = u.y, c = v.x, d = v.y;
  uint k1 = mul1(c, add1(a, b));
  uint k2 = mul1(a, sub1(d, c));
  uint k3 = mul1(b, neg(add1(d, c)));
  return U2(mod(k1 + k3), mod(k1 + k2));
}

// input, output 31 bits. Uses (a + i*b)^2 == ((a+b)*(a-b) + i*2*a*b).
uint2 sq(uint2 a) { return U2(mul1(add1(a.x, a.y), sub1(a.x, a.y)), mul1(a.x, mod(a.y << 1))); }

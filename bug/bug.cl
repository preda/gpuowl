typedef uint T;
typedef uint2 T2;

// Prime M(31) == 2^31 - 1
#define M31 0x7fffffffu

// make a pair of Ts.
T2 U2(T a, T b) { return (T2)(a, b); }

ulong u64(uint a) { return a; } // cast to 64 bits.

uint lo(ulong a) { return a & 0xffffffff; }
uint up(ulong a) { return a >> 32; }

// input 32 bits except 2^32-1; output 31 bits.
uint mod(uint x) { return x; }

// input 63 bits; output 31 bits.
uint bigmod(ulong x) {
  x = u64(2 * up(x)) + lo(x); // 'up' is 31 bits.
  return mod(2 * up(x) + lo(x));
}

// negative: -x; input 31 bits.
uint neg(uint x) { return M31 - x; }
  
uint add1(uint a, uint b) { return mod(a + b); }
uint sub1(uint a, uint b) { return add1(a, neg(b)); }

uint2 add(uint2 a, uint2 b) { return U2(add1(a.x, b.x), add1(a.y, b.y)); }
uint2 sub(uint2 a, uint2 b) { return U2(sub1(a.x, b.x), sub1(a.y, b.y)); }

// Input can be full 32bits. k <= 30 (i.e. mod 31).
uint shl1(uint a,  uint k) { return bigmod(u64(a) << k); }

ulong wideMul(uint a, uint b) { return u64(a) * b; }

// The main, complex multiplication; input and output 31 bits.
// (a + i*b) * (c + i*d) mod reduced.
uint2 mul(uint2 u, uint2 v) {
  uint a = u.x, b = u.y, c = v.x, d = v.y;
  ulong k1 = wideMul(c, add1(a, b));
  ulong k2 = wideMul(a, sub1(d, c));
  ulong k3 = wideMul(b, neg(add1(d, c)));
  // k1..k3 have at most 62 bits, so sums are at most 63 bits.
  return U2(bigmod(k1 + k3), bigmod(k1 + k2));
}

#define X2(a, b) { T2 t = a; a = add(t, b); b = sub(t, b); }

void fft4Core(T2 *u) {
  X2(u[0], u[2]);
  X2(u[1], u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);
}

void fft8Core(T2 *u) {
  for (int i = 0; i < 4; ++i) { X2(u[i], u[i + 4]); }
  fft4Core(u);
  fft4Core(u + 4);
}

void fft8(T2 *u) {
  fft8Core(u);
}

void bar()    { barrier(CLK_LOCAL_MEM_FENCE); }

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

void tabMul(const T2 *trig, T2 *u, uint n, uint f) {
  uint me = get_local_id(0);
  for (int i = 1; i < n; ++i) { u[i] = mul(u[i], trig[me / 32 + i * 8]); }
}

kernel __attribute__((reqd_work_group_size(256, 1, 1))) void bug(global T2 *io, global T2 *trig) {
  local uint lds[8 * 256];
  uint me = get_local_id(0);
  T2 u[8];
  for (int i = 0; i < 8; ++i) { u[i] = io[256 * i + me]; }
  fft8(u);
  shufl(lds, u, 8, 32);
  // bar();
  tabMul(trig, u, 8, 32);
  for (int i = 0; i < 8; ++i) { io[256 * i + me] = u[i]; }  
}

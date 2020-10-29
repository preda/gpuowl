typedef int i32;
typedef uint u32;
typedef long i64;
typedef ulong u64;

#if DEBUG
#define assert(condition) if (!(condition)) { printf("assert(%s) failed at line %d\n", STR(condition), __LINE__ - 1); }
// __builtin_trap();
#else
#define assert(condition)
//__builtin_assume(condition)
#endif // DEBUG

#define BIG_HEIGHT (SMALL_HEIGHT * MIDDLE)
#define ND (WIDTH * BIG_HEIGHT)
#define TRIG_STEP ((float) (-1.0 / ND))

global float4 TRIG[ND];
global float2 DELTA[ND];

#if WIDTH == 1024 || WIDTH == 256
#define NW 4
#else
#define NW 8
#endif

#if SMALL_HEIGHT == 1024 || SMALL_HEIGHT == 256
#define NH 4
#else
#define NH 8
#endif

#define G_W (WIDTH / NW)
#define G_H (SMALL_HEIGHT / NH)

#define OUT_WG 256
#define IN_WG  256
#define OUT_SIZEX 32
#define OUT_SPACING 4
#define IN_SIZEX 32
#define IN_SPACING 1

#define OVERLOAD __attribute__((overloadable))
#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

KERNEL(256) writeTrig(const global float4* in) {
  u32 gid = get_global_id(0);
  for (u32 n = 0; n < ND; n += get_global_size(0)) {
    TRIG[n + gid] = in[n + gid];
  }
}

KERNEL(256) writeDelta(const global float2* in) {
  u32 gid = get_global_id(0);
  for (u32 n = 0; n < ND; n += get_global_size(0)) {
    DELTA[n + gid] = in[n + gid];
  }
}

typedef float2 T;
typedef float4 T2;

// OVERLOAD double2 U2(double x, double y) { return (double2) (x, y); }
OVERLOAD T2 U2(T x, T y) { return (T2) (x, y); }
OVERLOAD T U2(float x, float y) { return (T) (x, y); }

float hwSin(float x) {
  float out;
  __asm("v_sin_f32_e32 %0, %1" : "=v"(out) : "v" (x));
  return out;
}

float hwCos(float x) {
  float out;
  __asm("v_cos_f32_e32 %0, %1" : "=v"(out) : "v" (x));
  return out;
}

T2 cosSin(u32 k) {
  assert(k < ND);
#if 0
  return TRIG[k];
#else
  float a = k * TRIG_STEP;
  float2 delta = DELTA[k];
  return (T2)(hwCos(a), delta.x, hwSin(a), delta.y);
#endif
}

KERNEL(256) readHwTrig(global float2* outCosSin) {
  u32 k = get_global_id(0);
  float a = k * TRIG_STEP;
  outCosSin[k] = (float2) (hwCos(a), hwSin(a));
}

// See Fast-Two-Sum and Two-Sum in
// "Extended-Precision Floating-Point Numbers for GPU Computation" by Andrew Thall

// Fast: assumes |a| >= |b|; cost: 3 ADD
float2 fastTwoSum(float a, float b) {
  float s = a + b;
  return U2(s, b - (s - a));
}

// cost: 6 ADD
float2 twoSum(float a, float b) {
  float s = a + b;
#if 0
  if (fabs(b) > fabs(a)) { float t = a; a = b; b = t; }
  float e = b - (s - a);
#elif 0
  float e = ((fabs(a) >= fabs(b)) ? b - (s - a) : (a - (s - b)));
#else
  float b1 = s - a;
  float a1 = s - b1;
  float e = (b - b1) + (a - a1);
#endif
  return U2(s, e);
}

// See Two-Product-FMA in the A. Thall paper above.
float2 twoMul(float a, float b) {
  float x = a * b;
  float e = fma(a, b, -x);
  return U2(x, e);
}

#define X2(a, b) { T2 t = a; a = sum(t, b); b = sum(t, -b); }
#define SWAP(a, b) { T2 t = a; a = b; b = t; }

// assumes |a| >= |b|. 14 ADD.
T fastSum(T a, T b) {
  T s = fastTwoSum(a.x, b.x);
  T t = fastTwoSum(a.y, b.y);
  s = fastTwoSum(s.x, s.y + t.x);
  s = fastTwoSum(s.x, s.y + t.y);                 
  return s;
}

OVERLOAD T sum(T a, T b) {
#if 1
  return (fabs(a.x) >= fabs(b.x)) ? fastSum(a, b) : fastSum(b, a);  
  // { T tmp = a; a = b; b = tmp; }  
#else
  // 20 ADD
  T s = twoSum(a.x, b.x);
  T t = twoSum(a.y, b.y);
  s = fastTwoSum(s.x, s.y + t.x);
  s = fastTwoSum(s.x, s.y + t.y);
  return s;
#endif
}

OVERLOAD T mul(T a, T b) {
  float2 t = twoMul(a.x, b.x);
  t.y = fma(a.x, b.y, fma(a.y, b.x, t.y));
  t = fastTwoSum(t.x, t.y);
  t.y = fma(a.y, b.y, t.y);
  // t = fastTwoSum(t.x, t.y);
  return t;
  // return U2(t.x, fma(a.y, b.y, fma(a.x, b.y, fma(a.y, b.x, t.y))));
}

OVERLOAD T sq(T a) {
  float2 t = twoMul(a.x, a.x);
  t.y = fma(a.x * 2, a.y, t.y);
  // t = fastTwoSum(t.x, t.y);
  t.y = fma(a.y, a.y, t.y);
  t = fastTwoSum(t.x, t.y);
  return t;
  // return U2(t.x, fma(a.y, a.y, fma(a.x * 2, a.y, t.y)));
}

// Complex sum
OVERLOAD T2 sum(T2 a, T2 b) { return U2(sum(a.xy, b.xy), sum(a.zw, b.zw)); }

// Complex mul
OVERLOAD T2 mul(T2 a, T2 b) {
  return U2(sum(mul(a.xy, b.xy), -mul(a.zw, b.zw)), sum(mul(a.xy, b.zw), mul(a.zw, b.xy)));
}

// Complex sq
OVERLOAD T2 sq(T2 a) { return U2(sum(sq(a.xy), -sq(a.zw)), mul(a.xy, a.zw) * 2); }

// Complex mul with real.
OVERLOAD T2 mul(T2 a, T b) {
  return U2(mul(a.xy, b), mul(a.zw, b));
}

// Complex multiply-add
OVERLOAD T2 mul(T2 a, T2 b, T2 c) { return sum(mul(a, b), c); }

#define M_SQRT1_2_FF (float2) (0.70710676908493041992f, 1.2101617485882343317e-08f)

T2 mul_t4(T2 a)  { return U2(a.zw, -a.xy); }
T2 mul_t8(T2 a)  { return mul(U2(sum(a.xy,  a.zw), sum(-a.xy, a.zw)),  M_SQRT1_2_FF); }
T2 mul_3t8(T2 a) { return mul(U2(sum(a.xy, -a.zw), sum( a.xy, a.zw)), -M_SQRT1_2_FF); }

void fft4Core(T2 *u) {
  X2(u[0], u[2]);
  X2(u[1], u[3]);
  u[3] = mul_t4(u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);
}

void fft4(T2 *u) {
  fft4Core(u);
  // revbin [0, 2, 1, 3] undo
  SWAP(u[1], u[2]);
}

void fft8Core(T2 *u) {
  X2(u[0], u[4]);
  X2(u[1], u[5]); u[5] = mul_t8(u[5]);
  X2(u[2], u[6]); u[6] = mul_t4(u[6]);
  X2(u[3], u[7]); u[7] = mul_3t8(u[7]);
  fft4Core(u);
  fft4Core(u + 4);  
}

void fft8(T2 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

void bar() { barrier(0); }

void shufl(u32 WG, local T2 *lds2, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  u32 m = me / f;
    
  local T* lds = (local T*) lds2;

  for (u32 i = 0; i < n; ++i) { lds[(m + i * WG / f) / n * f + m % n * WG + me % f] = u[i].xy; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].xy = lds[i * WG + me]; }
  bar();
  for (u32 i = 0; i < n; ++i) { lds[(m + i * WG / f) / n * f + m % n * WG + me % f] = u[i].zw; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].zw = lds[i * WG + me]; }
}

void shufl2(u32 WG, local T2 *lds2, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  local T* lds = (local T*) lds2;

  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);
  
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i].xy; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].xy = lds[i * WG + me]; }
  bar();
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i].zw; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].zw = lds[i * WG + me]; }
}

void tabMul(u32 WG, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);

  for (u32 i = 1; i < n; ++i) { u[i] = mul(u[i], cosSin(ND / WG / n * i * (me & ~(f - 1)))); }
}

void shuflAndMul(u32 WG, local T2 *lds, T2 *u, u32 n, u32 f) {
  shufl(WG, lds, u, n, f);
  tabMul(WG, u, n, f);
}

void shuflAndMul2(u32 WG, local T2 *lds, T2 *u, u32 n, u32 f) {
  tabMul(WG, u, n, f);
  shufl2(WG, lds, u, n, f);
}

// 64x4
void fft256w(local T2 *lds, T2 *u) {
  for (i32 s = 4; s >= 0; s -= 2) {
    if (s != 4) { bar(); }
    fft4(u);
    shuflAndMul(64, lds, u, 4, 1 << s);
  }
  fft4(u);
}

// 64x4
void fft256h(local T2 *lds, T2 *u) {
  u32 me = get_local_id(0);
  fft4(u);
  shuflAndMul2(64, lds, u, 4, 1);
  bar();
  fft4(u);
  shuflAndMul2(64, lds, u, 4, 4);
  bar();
  fft4(u);
  shuflAndMul2(64, lds, u, 4, 16);
  fft4(u);
}

// 256x4
void fft1Kw(local T2 *lds, T2 *u) {
  for (i32 s = 0; s <= 6; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul2(256, lds, u, 4, 1 << s);
  }
  fft4(u);
}

// 256x4
void fft1Kh(local T2 *lds, T2 *u) {
  fft4(u);
  shuflAndMul(256, lds, u, 4, 64);
  fft4(u);
  bar();
  shuflAndMul(256, lds, u, 4, 16);
  fft4(u);
  bar();
  shuflAndMul(256, lds, u, 4, 4);
  fft4(u);
  bar();
  shuflAndMul(256, lds, u, 4, 1);
  fft4(u);
}

// 512x8
void fft4Kw(local T2 *lds, T2 *u) {
  for (i32 s = 6; s >= 0; s -= 3) {
    if (s != 6) { bar(); }
    fft8(u);
    shuflAndMul(512, lds, u, 8, 1 << s);
  }
  fft8(u);
}

// 512x8
void fft4Kh(local T2 *lds, T2 *u) {
  fft8(u);
  shuflAndMul(512, lds, u, 8, 64);
  fft8(u);
  bar();
  shuflAndMul(512, lds, u, 8, 8);
  fft8(u);
  bar();
  shuflAndMul(512, lds, u, 8, 1);
  fft8(u);
}

void read(u32 WG, u32 N, T2 *u, const global T2 *in, u32 base) {
  for (u32 i = 0; i < N; ++i) { u[i] = in[base + i * WG + (u32) get_local_id(0)]; }
}

void write(u32 WG, u32 N, T2 *u, global T2 *out, u32 base) {
  for (u32 i = 0; i < N; ++i) { out[base + i * WG + (u32) get_local_id(0)] = u[i]; }
}

void fft_WIDTH(local T2 *lds, T2 *u) {
#if WIDTH == 256
  fft256w(lds, u);
#elif WIDTH == 512
  fft512w(lds, u);
#elif WIDTH == 1024
  fft1Kw(lds, u);
#elif WIDTH == 4096
  fft4Kw(lds, u);
#else
#error unexpected WIDTH.  
#endif  
}

void fft_HEIGHT(local T2 *lds, T2 *u) {
#if SMALL_HEIGHT == 256
  fft256h(lds, u);
#elif SMALL_HEIGHT == 512
  fft512h(lds, u);
#elif SMALL_HEIGHT == 1024
  fft1Kh(lds, u);
#elif SMALL_HEIGHT == 4096
  fft4Kh(lds, u);
#else
#error unexpected SMALL_HEIGHT.
#endif
}

void transposeWords(u32 W, u32 H, local T2 *lds, const T2 *in, T2 *out) {
  u32 GPW = W / 64, GPH = H / 64;

  u32 g = get_group_id(0);
  u32 gy = g % GPH;
  u32 gx = g / GPH;
  gx = (gy + gx) % GPW;

  in   += gy * 64 * W + gx * 64;
  out  += gy * 64     + gx * 64 * H;
  
  u32 me = get_local_id(0);
  u32 mx = me % 64;
  u32 my = me / 64;
  
  T2 u[16];

  for (i32 i = 0; i < 16; ++i) { u[i] = in[(4 * i + my) * W + mx]; }

  for (i32 i = 0; i < 16; ++i) {
    u32 l = i * 4 + me / 64;
    lds[l * 64 + (me + l) % 64 ] = u[i];
  }
  bar();
  for (i32 i = 0; i < 16; ++i) {
    u32 c = i * 4 + me / 64;
    u32 l = me % 64;
    u[i] = lds[l * 64 + (c + l) % 64];
  }

  for (i32 i = 0; i < 16; ++i) {
    out[(4 * i + my) * H + mx] = u[i];
  }
}

KERNEL(256) transposeOut(global T2* out, const global T2* in) {
  local T2 lds[4096];
  transposeWords(WIDTH, BIG_HEIGHT, lds, in, out);
}

KERNEL(256) transposeIn(global T2* out, const global T2* in) {
  local T2 lds[4096];
  transposeWords(BIG_HEIGHT, WIDTH, lds, in, out);
}

void middleMul2(T2 *u, u32 g, u32 me) {
#if MIDDLE != 1
#error middle
#endif
  
  u[0] = mul(u[0], cosSin(g * me));
}

void middleShuffle(local T2 *lds, T2 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  lds[(me % blockSize) * (workgroupSize / blockSize) + me / blockSize] = u[0];
  bar();
  u[0] = lds[me];
}

KERNEL(IN_WG) fftMiddleIn(global T2* out, const global T2* in) {
  T2 u[MIDDLE];
  
  u32 SIZEY = IN_WG / IN_SIZEX;

  u32 N = WIDTH / IN_SIZEX;
  
  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % IN_SIZEX;
  u32 my = me / IN_SIZEX;

  u32 startx = gx * IN_SIZEX;
  u32 starty = gy * SIZEY;

  in += starty * WIDTH + startx;
  u[0] = in[my * WIDTH + mx];

  middleMul2(u, startx + mx, starty + my);

  // fft_MIDDLE(u);
  // middleMul(u, starty + my);
  
  local T2 lds[IN_WG * MIDDLE];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);

  out += gx * (MIDDLE * SMALL_HEIGHT * IN_SIZEX) + (gy / IN_SPACING) * (MIDDLE * IN_WG * IN_SPACING) + (gy % IN_SPACING) * SIZEY;
  out += (me / SIZEY) * (IN_SPACING * SIZEY) + (me % SIZEY);

  for (i32 i = 0; i < MIDDLE; ++i) { out[i * (IN_WG * IN_SPACING)] = u[i]; }
}

KERNEL(OUT_WG) fftMiddleOut(global T2* out, global T2* in) {
  T2 u[MIDDLE];

  u32 SIZEY = OUT_WG / OUT_SIZEX;

  u32 N = SMALL_HEIGHT / OUT_SIZEX;

  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % OUT_SIZEX;
  u32 my = me / OUT_SIZEX;

  // Kernels read OUT_SIZEX consecutive T2.
  // Each WG-thread kernel processes OUT_SIZEX columns from a needed SMALL_HEIGHT columns
  // Each WG-thread kernel processes SIZEY rows out of a needed WIDTH rows

  u32 startx = gx * OUT_SIZEX;  // Each input column increases FFT element by one
  u32 starty = gy * SIZEY;  // Each input row increases FFT element by BIG_HEIGHT
  in += starty * BIG_HEIGHT + startx;

  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = in[i * SMALL_HEIGHT + my * BIG_HEIGHT + mx]; }
  // middleMul(u, startx + mx);
  // fft_MIDDLE(u);

  middleMul2(u, starty + my, startx + mx);
  local T2 lds[OUT_WG * MIDDLE];

  middleShuffle(lds, u, OUT_WG, OUT_SIZEX);

  out += gx * (MIDDLE * WIDTH * OUT_SIZEX);
  out += (gy / OUT_SPACING) * (MIDDLE * (OUT_WG * OUT_SPACING));
  out += (gy % OUT_SPACING) * SIZEY;
  out += (me / SIZEY) * (OUT_SPACING * SIZEY);
  out += (me % SIZEY);

  for (i32 i = 0; i < MIDDLE; ++i) { out[i * (OUT_WG * OUT_SPACING)] = u[i]; }
}

KERNEL(G_W) fftWin(global T2* out, global const T2* in) {
  local T2 lds[WIDTH / 2];
  T2 u[NW];
  u32 g = get_group_id(0);
  read(G_W, NW, u, in, WIDTH * g);
  fft_WIDTH(lds, u);
  write(G_W, NW, u, out, WIDTH * g);
}

void readCarryFusedLine(const global T2* in, T2 *u, u32 line) {
  u32 me = get_local_id(0);
  u32 WG = OUT_WG * OUT_SPACING;
  u32 SIZEY = WG / OUT_SIZEX;

  in += line % OUT_SIZEX * SIZEY + line % SMALL_HEIGHT / OUT_SIZEX * WIDTH / SIZEY * MIDDLE * WG + line / SMALL_HEIGHT * WG;
  in += me / SIZEY * MIDDLE * WG + me % SIZEY;
  for (i32 i = 0; i < NW; ++i) { u[i] = in[i * G_W / SIZEY * MIDDLE * WG]; }
}

KERNEL(G_W) fftWout(global T2* out, global const T2* in) {
  local T2 lds[WIDTH / 2];
  T2 u[NW];
  u32 g = get_group_id(0);
  readCarryFusedLine(in, u, g);
  fft_WIDTH(lds, u);
  write(G_W, NW, u, out, WIDTH * g);
}

u32 transPos(u32 k, u32 width, u32 height) { return k / height + k % height * width; }

void readTailFusedLine(const global T2* in, T2 *u, u32 line, u32 memline) {
  // We go to some length here to avoid dividing by MIDDLE in address calculations.
  // The transPos converted logical line number into physical memory line numbers
  // using this formula:  memline = line / WIDTH + line % WIDTH * MIDDLE.
  // We can compute the 0..9 component of address calculations as line / WIDTH,
  // and the 0,10,20,30,..310 component as (line % WIDTH) % 32 = (line % 32),
  // and the multiple of 320 component as (line % WIDTH) / 32

  u32 me = get_local_id(0);
  u32 WG = IN_WG * IN_SPACING;
  u32 SIZEY = WG / IN_SIZEX;

  in += line / WIDTH * WG;
  in += line % IN_SIZEX * SIZEY;
  in += line % WIDTH / IN_SIZEX * (SMALL_HEIGHT / SIZEY) * MIDDLE * WG;
  in += me / SIZEY * MIDDLE * WG + me % SIZEY;
  for (i32 i = 0; i < NH; ++i) { u[i] = in[i * G_H / SIZEY * MIDDLE * WG]; }
}

KERNEL(G_H) fftHin(global T2* out, const global T2* in) {
  local T2 lds[SMALL_HEIGHT / 2];
  T2 u[NH];
  u32 g = get_group_id(0);
  readTailFusedLine(in, u, g, transPos(g, MIDDLE, WIDTH));
  fft_HEIGHT(lds, u);
  write(G_H, NH, u, out, SMALL_HEIGHT * transPos(g, MIDDLE, WIDTH));
}

KERNEL(G_H) fftHout(global T2* io) {
  local T2 lds[SMALL_HEIGHT / 2];
  T2 u[NH];
  u32 g = get_group_id(0);
  io += g * SMALL_HEIGHT;
  read(G_H, NH, u, io, 0);
  fft_HEIGHT(lds, u);
  write(G_H, NH, u, io, 0);
}

T2 addsub(T2 a) { return U2(sum(a.xy, a.zw), sum(a.xy, -a.zw)); }

// computes 2*(a.x*b.x+a.y*b.y) + i*2*(a.x*b.y+a.y*b.x)
T2 foo2(T2 a, T2 b) {
  a = addsub(a);
  b = addsub(b);
  return addsub(U2(mul(a.xy, b.xy), mul(a.zw, b.zw)));
}

// computes 2*[x^2+y^2 + i*(2*x*y)]. Needs a name.
T2 foo(T2 a) {
  a = addsub(a);
  return addsub(U2(sq(a.xy), sq(a.zw)));
}

T2 conjugate(T2 a) { return U2(a.xy, -a.zw); }

KERNEL(SMALL_HEIGHT / 2) square(global T2* io) {
  uint W = SMALL_HEIGHT;
  uint H = ND / W;

  uint line1 = get_group_id(0);  
  uint me = get_local_id(0);

  if (line1 == 0 && me == 0) {
    io[0]     = foo(conjugate(io[0])) * 2;  // MUL by powers of 2 is simple
    io[W / 2] = sq(conjugate(io[W / 2])) * 4;
    return;
  }

  uint line2 = (H - line1) % H;
  uint g1 = transPos(line1, MIDDLE, WIDTH);
  uint g2 = transPos(line2, MIDDLE, WIDTH);
  uint k = g1 * W + me;
  uint v = g2 * W + (W - 1) - me + (line1 == 0);

  T2 a = io[k];
  T2 b = conjugate(io[v]);

  T2 t = - cosSin(me * H + line1);
  
  X2(a, b);
  T2 a2 = sq(a);
  T2 b2 = sq(b);
  b = mul(a, b) * 2;
  a = mul(b2, t, a2);
  X2(a, b);
  
  io[k] = conjugate(a);
  io[v] = b;
}

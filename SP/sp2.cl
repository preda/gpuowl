// Outside forces should prepend a precomputed table of trig delta relative to HW trig,
// a pair of floats being stored as one u64.
// global u64 trigDelta[] = {...};

typedef int i32;
typedef uint u32;
typedef long i64;
typedef ulong u64;

typedef float2 T;
typedef float4 T2;

#define OVERLOAD __attribute__((overloadable))
#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

// #define TRIG_STEP (-1.0f / ND)

float hwSin(float x) {
  float out;
  __asm volatile ("v_sin_f32_e32 %0, %1" : "=v"(out) : "v" (x));
  return out;
}

float hwCos(float x) {
  float out;
  __asm volatile ("v_cos_f32_e32 %0, %1" : "=v"(out) : "v" (x));
  return out;
}

// Return cos on the first pair of floats and sin on the next pair of floats.
float4 cosSin(u32 k) {
  float a = k * TRIG_STEP;
  float2 delta = as_float2(trigDelta[k]);
  return (float4) (hwCos(a), delta.x, hwSin(a), delta.y);
}

double toD(float2 a) { return a.x + (double) a.y; }

OVERLOAD double2 U2(double x, double y) { return (double2) (x, y); }
OVERLOAD T2 U2(T x, T y) { return (T2) (x, y); }
OVERLOAD T U2(float x, float y) { return (T) (x, y); }

// Return cos/sin as a pair of doubles.
double2 cosSinD(u32 k) {
  float4 cs = cosSin(k);
  return U2(toD(cs.xy), toD(cs.zw));
}

KERNEL(256) trig(global double2* out) {
  u32 k = get_global_id(0);
  out[k] = cosSinD(k);
}

KERNEL(256) readHwTrig(global float2* outCosSin) {
  u32 k = get_global_id(0);
  float a = k * TRIG_STEP;
  outCosSin[k] = (float2) (hwCos(a), hwSin(a));
}

// ----


#define X2(a, b) { T2 t = a; a = t + b; b = t - b; }
#define SWAP(a, b) { T2 t = a; a = b; b = t; }

T sum(T a, T b) {
  float s = a.x + b.x;
  float c = s - a.x;
  return U2(s, a.y + b.y + (b.x - c));
}

T mul(T a, T b) {
  float c = a.x * b.x;
  float d = fma(a.x, b.x, -c);
  // d = fma(a.y, b.y, d); // optional
  d = fma(a.x, b.y, d);
  d = fma(a.y, b.x, d);
  return U2(c, d);
}

// Complex mul
T2 mul(T2 a, T2 b) {
  T c = mul(a.xy, b.xy);
  T d = mul(a.zw, b.zw);
  T e = mul(a.xy, b.zw);
  T f = mul(a.zw, b.xy);
  return U2(sum(c, -d), sum(e, f));
}

T2 mul_t4(T2 a) { return U2(a.zw, -a.xy); }
T2 mul_t8(T2 a) { return U2(a.zw + a.xy, a.zw - a.xy) * M_SQRT1_2_F; }
T2 mul_3t8(T2 a) { return U2(a.xy - a.zw, a.xy + a.zw) * - M_SQRT1_2_F; }

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

void tabMul(u32 WG, const global T2 *trig, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  for (i32 i = 1; i < n; ++i) {
    // u[i] = mul(u[i], trig[me / f + i * (WG / f)]); }
    // cosSin((ND / WG) * i * (me / f) * f);
    u[i] = mul(u[i], cosSin((ND / WG) * i * (me & ~(f - 1))));
  }
}


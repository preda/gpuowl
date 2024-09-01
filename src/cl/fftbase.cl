// Copyright (C) Mihai Preda

#include "fft4.cl"
#include "fft8.cl"
#include "trig.cl"
// #include "math.cl"

void shufl(u32 WG, local T2 *lds2, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);
  local T* lds = (local T*) lds2;

  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i].x; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].x = lds[i * WG + me]; }
  bar();
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i].y; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].y = lds[i * WG + me]; }
}

void tabMul(u32 WG, Trig trig, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  u32 p = me & ~(f - 1);
  T2 w = trig[p];

  if (n >= 8) {
    u[1] = cmulFancy(u[1], w);
  } else {
    u[1] = cmul(u[1], w);
  }

#if CLEAN == 1
  T2 base = trig[WG + p];

  if (n >= 8) {
    for (u32 i = 2; i < n; ++i) {
      u[i] = cmul(u[i], base);
      base = cmulFancy(base, w);
    }
  } else {
    for (u32 i = 2; i < n; ++i) {
      u[i] = cmul(u[i], base);
      base = cmul(base, w);
    }
  }

#elif CLEAN == 0
  if (n >= 8) {
    T a = 2 * fma(w.x, w.y, w.y); // 2*sin*cos
    u[2] = cmulFancy(u[2], U2(-2 * w.y * w.y, a));
    a *= 2;
    T2 base = U2(fma(a, -w.y, w.x + 1), fma(a, w.x, a - w.y));
    for (u32 i = 3; i < n; ++i) {
      u[i] = cmul(u[i], base);
      base = cmulFancy(base, w);
    }
  } else {
    T a = 2 * w.x * w.y;
    // u[2] = fancyMulTrig(u[2], U2(-2 * w.y * w.y, a));
    // u[2] = mul(u[2], U2(fma(w.x, w.x, -w.y * w.y), a));
    u[2] = cmul(u[2], U2(fma(-2 * w.y, w.y, 1), a));
    a *= 2;
    T2 base = U2(fma(a, -w.y, w.x), fma(a, w.x, -w.y));
    for (u32 i = 3; i < n; ++i) {
      u[i] = cmul(u[i], base);
      base = cmul(base, w);
    }
  }
#else
#error CLEAN must be 0 or 1
#endif
}

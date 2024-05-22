// Copyright (C) Mihai Preda

#include "fft4.cl"
#include "fft8.cl"
// #include "math.cl"

void shufl(u32 WG, local T2 *lds2, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  local T* lds = (local T*) lds2;

  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);

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
  if (n >= 8) {
    u[1] = fancyMulTrig(u[1], trig[p]);
  } else {
    u[1] = mul(u[1], trig[p]);
  }
  for (u32 i = 2; i < n; ++i) {
    u[i] = mul(u[i], trig[p + WG * (i - 1)]);
  }
}

void shuflAndMul(u32 WG, local T2 *lds, Trig trig, T2 *u, u32 n, u32 f) {
  tabMul(WG, trig, u, n, f);
  shufl(WG, lds, u, n, f);
}

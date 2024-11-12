// Copyright (C) Mihai Preda

#include "fftbase.cl"

#if WIDTH != 256 && WIDTH != 512 && WIDTH != 1024 && WIDTH != 4096 && WIDTH != 625
#error WIDTH must be one of: 256, 512, 1024, 4096, 625
#endif

void fft_NW(T2 *u) {
#if NW == 4
  fft4(u);
#elif NW == 5
  fft5(u);
#elif NW == 8
  fft8(u);
#else
#error NW
#endif
}

#if 0 && NW == 4 && WIDTH == 1024
int bcast4(int x)  { return __builtin_amdgcn_mov_dpp(x, 0, 0xf, 0xf, false); }
int bcast16(int x) { return __builtin_amdgcn_ds_swizzle(x, 0x0010); }
int bcast64(int x) { return __builtin_amdgcn_readfirstlane(x); }

int bcastAux(int x, u32 span) {
  return span == 4 ? bcast4(x) : span == 16 ? bcast16(x) : span == 64 ? bcast64(x) : x;
}

T2 bcast(T2 src, u32 span) {
  int4 s = as_int4(src);
  for (int i = 0; i < 4; ++i) { s[i] = bcastAux(s[i], span); }
  return as_double2(s);
}

void chainMul(T2 *u, u32 n, T2 w) {
  T2 base = w;
  for (int i = 1; i < n; ++i) {
    u[i] = cmul(u[i], w);
    w = cmul(w, base);
  }
}

void fft_WIDTH(local T2 *lds, T2 *u, Trig trig) {
  u32 me = get_local_id(0);
  T2 w = slowTrig_N(ND / WIDTH * me, ND / NW); // trig[me];

#if !UNROLL_W
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WIDTH / NW; s *= NW) {
    if (s > 1) { bar(); }
    fft_NW(u);
    w = bcast(w, s);
    chainMul(u, NW, w);
    shufl( WIDTH / NW, lds,  u, NW, s);
  }
  fft_NW(u);
}

#else

void fft_WIDTH(local T2 *lds, T2 *u, Trig trig) {

#if !UNROLL_W
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WIDTH / NW; s *= NW) {
    if (s > 1) { bar(); }
    fft_NW(u);
    tabMul(WIDTH / NW, trig, u, NW, s);
    shufl( WIDTH / NW, lds,  u, NW, s);
  }
  fft_NW(u);
}

#endif

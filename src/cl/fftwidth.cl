// Copyright (C) Mihai Preda

#include "fftbase.cl"

void fft_NW(T2 *u) {
#if NW == 4
  fft4(u);
#elif NW == 8
  fft8(u);
#else
#error NW
#endif
}

#if 0
T2 swizzle(T2 src) {
  int4 s = as_int4(src);
  if (NW == 4) {
    const int how = 0x1c;
    s.x = __builtin_amdgcn_ds_swizzle(s.x, how);
    s.y = __builtin_amdgcn_ds_swizzle(s.y, how);
    s.z = __builtin_amdgcn_ds_swizzle(s.z, how);
    s.w = __builtin_amdgcn_ds_swizzle(s.w, how);
  } else {
    const int how = 0x18;
    s.x = __builtin_amdgcn_ds_swizzle(s.x, how);
    s.y = __builtin_amdgcn_ds_swizzle(s.y, how);
    s.z = __builtin_amdgcn_ds_swizzle(s.z, how);
    s.w = __builtin_amdgcn_ds_swizzle(s.w, how);
  }
  return as_double2(s);
}
#endif

void fft_WIDTH(local T2 *lds, T2 *u, Trig trig) {

#if WIDTH != 256 && WIDTH != 512 && WIDTH != 1024 && WIDTH != 4096
#error WIDTH must be one of: 256, 512, 1024, 4096
#endif

  UNROLL_WIDTH_CONTROL
  for (u32 s = 1; s < WIDTH / NW; s *= NW) {
    if (s > 1) { bar(); }
    fft_NW(u);
    tabMul(WIDTH / NW, trig, u, NW, s);
    shufl( WIDTH / NW, lds,  u, NW, s);
  }
  fft_NW(u);
}

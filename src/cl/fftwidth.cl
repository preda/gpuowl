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

void fft_WIDTH(local T2 *lds, T2 *u, Trig trig) {

#if WIDTH != 256 && WIDTH != 512 && WIDTH != 1024 && WIDTH != 4096
#error WIDTH must be one of: 256, 512, 1024, 4096
#endif

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

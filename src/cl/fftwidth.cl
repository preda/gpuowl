// Copyright (C) Mihai Preda

#include "fftbase.cl"

#if WIDTH != 256 && WIDTH != 512 && WIDTH != 1024 && WIDTH != 4096 && WIDTH != 625
#error WIDTH must be one of: 256, 512, 1024, 4096, 625
#endif

void fft_NW(T2 *u, bool initialX2done) {
#if NW == 4
  if (initialX2done) fft4initialX2done(u);
  else fft4(u);
#elif NW == 5
  fft5(u);
#elif NW == 8
  if (initialX2done) fft8initialX2done(u);
  else fft8(u);
#else
#error NW
#endif
}

#if BCAST && (WIDTH <= 1024)

void fft_WIDTH(local T2 *lds, T2 *u, Trig trig, bool initialX2done) {
  u32 me = get_local_id(0);
#if NW == 8
  T2 w = fancyTrig_N(ND / WIDTH * me);
#else
  T2 w = slowTrig_N(ND / WIDTH * me, ND / NW);
#endif

  if (initialX2done) fft_NW(u, initialX2done);
  for (u32 s = 1; s < WIDTH / NW; s *= NW) {
    if (initialX2done) {
      if (s > 1) { bar(); fft_NW(u, false); }
    } else {
      if (s > 1) bar();
      fft_NW(u, false);
    }
    w = bcast(w, s);

#if NW == 8
    chainMul8(u, w);
#else
    chainMul4(u, w);
#endif

    shufl( WIDTH / NW, lds,  u, NW, s);
  }
  fft_NW(u, false);
}

#else

void fft_WIDTH(local T2 *lds, T2 *u, Trig trig, bool initialX2done) {
  u32 me = get_local_id(0);

  if (initialX2done) fft_NW(u, initialX2done);
#if !UNROLL_W
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WIDTH / NW; s *= NW) {
    if (initialX2done) {
      if (s > 1) { bar(); fft_NW(u, false); }
    } else {
      if (s > 1) bar();
      fft_NW(u, false);
    }
    tabMul(WIDTH / NW, trig, u, NW, s, me);
    shufl( WIDTH / NW, lds,  u, NW, s);
  }
  fft_NW(u, false);
}

#endif

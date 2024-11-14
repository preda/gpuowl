// Copyright (C) Mihai Preda

#include "base.cl"
#include "fftbase.cl"
#include "middle.cl"

#if SMALL_HEIGHT != 256 && SMALL_HEIGHT != 512 && SMALL_HEIGHT != 1024 && SMALL_HEIGHT != 4096
#error SMALL_HEIGHT must be one of: 256, 512, 1024, 4096
#endif

u32 transPos(u32 k, u32 middle, u32 width) { return k / width + k % width * middle; }

void fft_NH(T2 *u) {
#if NH == 4
  fft4(u);
#elif NH == 8
  fft8(u);
#else
#error NH
#endif
}

#if BCAST && (HEIGHT <= 1024)

void fft_HEIGHT(local T2 *lds, T2 *u, Trig trig, T2 w) {
  // u32 me = get_local_id(0);
  // T2 w = slowTrig_N(ND / SMALL_HEIGHT * me, ND / NH);

  /*
#if !UNROLL_H
  __attribute__((opencl_unroll_hint(1)))
#endif
*/

  for (u32 s = 1; s < SMALL_HEIGHT / NH; s *= NH) {
    if (s > 1) { bar(); }
    fft_NH(u);
    w = bcast(w, s);

#if NH == 8
    chainMul8(u, w);
#else
    chainMul4(u, w);
#endif

    shufl(SMALL_HEIGHT / NH, lds,  u, NH, s);
  }
  fft_NH(u);
}

#else

void fft_HEIGHT(local T2 *lds, T2 *u, Trig trig, T2 w) {

#if !UNROLL_H
  __attribute__((opencl_unroll_hint(1)))
#endif

  for (u32 s = 1; s < SMALL_HEIGHT / NH; s *= NH) {
    if (s > 1) { bar(); }
    fft_NH(u);
    tabMul(SMALL_HEIGHT / NH, trig, u, NH, s);
    shufl(SMALL_HEIGHT / NH, lds,  u, NH, s);
  }
  fft_NH(u);
}

#endif

// Copyright (C) Mihai Preda and George Woltman

#include "base.cl"
#include "math.cl"
#include "fft-middle.cl"
#include "middle.cl"

KERNEL(OUT_WG) fftMiddleOut(P(T2) out, P(T2) in, Trig trig) {
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

  u32 x = startx + mx;
  u32 y = starty + my;

  readMiddleOutLine(u, in, y, x);

  middleMul(u, x, trig);

  fft_MIDDLE(u);

  // FFT results come out multiplied by the FFT length (NWORDS).  Also, for performance reasons
  // weights and invweights are doubled meaning we need to divide by another 2^2 and 2^2.
  // Finally, roundoff errors are sometimes improved if we use the next lower double precision
  // number.  This may be due to roundoff errors introduced by applying inexact TWO_TO_N_8TH weights.
  double factor = 1.0 / (4 * 4 * NWORDS);

  middleMul2(u, y, x, factor, trig);

#if MIDDLE_OUT_LDS_TRANSPOSE
  // Transpose the x and y values
  local T lds[OUT_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, OUT_WG, OUT_SIZEX);
  out += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  out += mx * SIZEY + my;
#endif

  writeMiddleOutLine(out, u, gy, gx);
}

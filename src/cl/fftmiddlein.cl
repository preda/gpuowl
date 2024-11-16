// Copyright (C) Mihai Preda and George Woltman

#include "base.cl"
#include "math.cl"
#include "fft-middle.cl"
#include "middle.cl"

KERNEL(IN_WG) fftMiddleIn(P(T2) out, CP(T2) in, Trig trig) {
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

  u32 x = startx + mx;
  u32 y = starty + my;

  readRotatedWidth(u, in, y, x);

  middleMul2(u, x, y, 1, trig);

  fft_MIDDLE(u);

  middleMul(u, y, trig);

#if !MIDDLE_SHUFFLE_WRITE
  local T lds[IN_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, IN_WG, IN_SIZEX);
  write(IN_WG, MIDDLE, u, out, gx * (BIG_HEIGHT * IN_SIZEX) + gy * (MIDDLE * IN_WG));
#else
  out += gx * (BIG_HEIGHT * IN_SIZEX) + gy * (MIDDLE * IN_WG);
  middleShuffleWrite(out, u, IN_WG, IN_SIZEX);
#endif

}

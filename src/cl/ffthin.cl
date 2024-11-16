// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"
#include "fftheight.cl"

// Do an FFT Height after a transposeW (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHin(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[SMALL_HEIGHT / 2];
  
  T2 u[NH];
  u32 g = get_group_id(0);

  readTailFusedLine(in, u, g);

  u32 me = get_local_id(0);
#if NH == 8
  T2 w = fancyTrig_N(ND / SMALL_HEIGHT * me);
#else
  T2 w = slowTrig_N(ND / SMALL_HEIGHT * me, ND / NH);
#endif

  fft_HEIGHT(lds, u, smallTrig, w);

  write(G_H, NH, u, out, SMALL_HEIGHT * transPos(g, MIDDLE, WIDTH));
}

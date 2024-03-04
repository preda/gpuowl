// Copyright (C) Mihai Preda

#include "gpuowl.cl"
#include "fftwidth.cl"

// Do an fft_WIDTH after a transposeH (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_W) fftW(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[WIDTH / 2];
  
  T2 u[NW];
  u32 g = get_group_id(0);

  readCarryFusedLine(in, u, g);
  fft_WIDTH(lds, u, smallTrig);  
  out += WIDTH * g;
  write(G_W, NW, u, out, 0);
}

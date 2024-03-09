// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"
#include "weight.cl"
#include "fftwidth.cl"

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig, BigTab THREAD_WEIGHTS) {
  local T2 lds[WIDTH / 2];

  T2 u[NW];
  u32 g = get_group_id(0);

  u32 step = WIDTH * g;
  in  += step;
  out += step;

  u32 me = get_local_id(0);

  T base = optionalHalve(fancyMul(THREAD_WEIGHTS[G_W + g / CARRY_LEN].y, THREAD_WEIGHTS[me].y));
  base = optionalHalve(fancyMul(base, fweightUnitStep(g % CARRY_LEN)));

  for (u32 i = 0; i < NW; ++i) {
    T w1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    T w2 = optionalHalve(fancyMul(w1, WEIGHT_STEP));
    u32 p = G_W * i + me;
    u[i] = U2(in[p].x, in[p].y) * U2(w1, w2);
  }

  fft_WIDTH(lds, u, smallTrig);
  
  write(G_W, NW, u, out, 0);
}

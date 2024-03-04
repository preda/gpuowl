// Copyright (C) Mihai Preda

#include "gpuowl.cl"
#include "trig.cl"
#include "onepairmul.cl"

// From original code t = swap(base) and we need sq(conjugate(t)).  This macro computes sq(conjugate(t)) from base^2.
#define swap_squared(a) (-a)

KERNEL(SMALL_HEIGHT / 2) kernelMultiply(P(T2) io, CP(T2) in, BigTab TRIG_BHW) {
  u32 W = SMALL_HEIGHT;
  u32 H = ND / W;

  u32 line1 = get_group_id(0);
  u32 me = get_local_id(0);

  if (line1 == 0 && me == 0) {
    io[0]     = foo2_m2(conjugate(io[0]), conjugate(in[0]));
    io[W / 2] = conjugate(mul_m4(io[W / 2], in[W / 2]));
    return;
  }

  u32 line2 = (H - line1) % H;
  u32 g1 = transPos(line1, MIDDLE, WIDTH);
  u32 g2 = transPos(line2, MIDDLE, WIDTH);
  u32 k = g1 * W + me;
  u32 v = g2 * W + (W - 1) - me + (line1 == 0);
  T2 a = io[k];
  T2 b = io[v];
  T2 c = in[k];
  T2 d = in[v];
  onePairMul(a, b, c, d, swap_squared(slowTrig_N(me * H + line1, ND / 2, TRIG_BHW)));
  io[k] = a;
  io[v] = b;
}

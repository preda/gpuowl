// Copyright (C) Mihai Preda

#include "gpuowl.cl"

//{{ TAIL_FUSED_MUL
#if MUL_2LOW
KERNEL(G_H) NAME(P(T2) out, CP(T2) in, Trig smallTrig2) {
#else
KERNEL(G_H) NAME(P(T2) out, CP(T2) in, CP(T2) a,
#if MUL_DELTA
                 CP(T2) b,
#endif
                 Trig smallTrig1, Trig smallTrig2) {
  // The arguments smallTrig1, smallTrig2 point to the same data; they are passed in as two buffers instead of one
  // in order to work-around the ROCm optimizer which would otherwise "cache" the data once read into VGPRs, leading
  // to poor occupancy.
#endif
  
  local T2 lds[SMALL_HEIGHT / 2];

  T2 u[NH], v[NH];
  T2 p[NH], q[NH];

  u32 W = SMALL_HEIGHT;
  u32 H = ND / W;

  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);
    
#if MUL_DELTA
  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  readDelta(G_H, NH, p, a, b, memline1 * SMALL_HEIGHT);
  readDelta(G_H, NH, q, a, b, memline2 * SMALL_HEIGHT);
  fft_HEIGHT(lds, u, smallTrig1);
  bar();
  fft_HEIGHT(lds, v, smallTrig1);
#elif MUL_LOW
  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  read(G_H, NH, p, a, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, a, memline2 * SMALL_HEIGHT);
  fft_HEIGHT(lds, u, smallTrig1);
  bar();
  fft_HEIGHT(lds, v, smallTrig1);
#elif MUL_2LOW
  read(G_H, NH, u, out, memline1 * SMALL_HEIGHT);
  read(G_H, NH, v, out, memline2 * SMALL_HEIGHT);
  read(G_H, NH, p, in, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, in, memline2 * SMALL_HEIGHT);
#else
  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  readTailFusedLine(a, p, line1);
  readTailFusedLine(a, q, line2);
  fft_HEIGHT(lds, u, smallTrig1);
  bar();
  fft_HEIGHT(lds, v, smallTrig1);
  bar();
  fft_HEIGHT(lds, p, smallTrig1);
  bar();
  fft_HEIGHT(lds, q, smallTrig1);
#endif

  u32 me = get_local_id(0);
  if (line1 == 0) {
    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, slowTrig_2SH(2 * me, SMALL_HEIGHT / 2), true);
    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);

    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, slowTrig_2SH(1 + 2 * me, SMALL_HEIGHT / 2), false);
    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
  } else {    
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);
    pairMul(NH, u, v, p, q, slowTrig_N(line1 + me * H, ND / 4), false);
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);
  }

  bar();
  fft_HEIGHT(lds, v, smallTrig2);
  write(G_H, NH, v, out, memline2 * SMALL_HEIGHT);

  bar();
  fft_HEIGHT(lds, u, smallTrig2);
  write(G_H, NH, u, out, memline1 * SMALL_HEIGHT);
}
//}}

//== TAIL_FUSED_MUL NAME=tailMulLowLow,   MUL_DELTA=0, MUL_LOW=0, MUL_2LOW=1
//== TAIL_FUSED_MUL NAME=tailFusedMulLow, MUL_DELTA=0, MUL_LOW=1, MUL_2LOW=0
//== TAIL_FUSED_MUL NAME=tailFusedMul,    MUL_DELTA=0, MUL_LOW=0, MUL_2LOW=0

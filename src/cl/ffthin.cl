// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"
#include "fftheight.cl"

#if FFT_FP64

// Do an FFT Height after an fftMiddleIn (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHin(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[SMALL_HEIGHT / 2];
  
  T2 u[NH];
  u32 g = get_group_id(0);

  u32 me = get_local_id(0);

  readTailFusedLine(in, u, g, me);

#if NH == 8
  T2 w = fancyTrig_N(ND / SMALL_HEIGHT * me);
#else
  T2 w = slowTrig_N(ND / SMALL_HEIGHT * me, ND / NH);
#endif

  fft_HEIGHT(lds, u, smallTrig, w);

  write(G_H, NH, u, out, SMALL_HEIGHT * transPos(g, MIDDLE, WIDTH));
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// Do an FFT Height after an fftMiddleIn (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHin(P(T2) out, CP(T2) in, Trig smallTrig) {
  local F2 lds[SMALL_HEIGHT / 2];

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;

  F2 u[NH];
  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  readTailFusedLine(inF2, u, g, me);

#if NH == 8
  F2 w = fancyTrig_N(ND / SMALL_HEIGHT * me);
#else
  F2 w = slowTrig_N(ND / SMALL_HEIGHT * me, ND / NH);
#endif

  fft_HEIGHT(lds, u, smallTrigF2);

  write(G_H, NH, u, outF2, SMALL_HEIGHT * transPos(g, MIDDLE, WIDTH));
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

// Do an FFT Height after an fftMiddleIn (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHinGF31(P(T2) out, CP(T2) in, Trig smallTrig) {
  local GF31 lds[SMALL_HEIGHT / 2];

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTHTRIGGF31);

  GF31 u[NH];
  u32 g = get_group_id(0);

  u32 me = get_local_id(0);

  readTailFusedLine(in31, u, g, me);

  fft_HEIGHT(lds, u, smallTrig31);

  write(G_H, NH, u, out31, SMALL_HEIGHT * transPos(g, MIDDLE, WIDTH));
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

// Do an FFT Height after an fftMiddleIn (which may not have fully transposed data, leading to non-sequential input)
KERNEL(G_H) fftHinGF61(P(T2) out, CP(T2) in, Trig smallTrig) {
  local GF61 lds[SMALL_HEIGHT / 2];

  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTHTRIGGF61);

  GF61 u[NH];
  u32 g = get_group_id(0);

  u32 me = get_local_id(0);

  readTailFusedLine(in61, u, g, me);

  fft_HEIGHT(lds, u, smallTrig61);

  write(G_H, NH, u, out61, SMALL_HEIGHT * transPos(g, MIDDLE, WIDTH));
}

#endif

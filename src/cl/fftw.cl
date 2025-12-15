// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"
#include "fftwidth.cl"
#include "middle.cl"

#if FFT_FP64

// Do the ending fft_WIDTH after an fftMiddleOut.  This is the same as the first half of carryFused.
KERNEL(G_W) fftW(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[WIDTH / 2];

  T2 u[NW];
  u32 g = get_group_id(0);

  readCarryFusedLine(in, u, g);
  fft_WIDTH(lds, u, smallTrig);  
  out += WIDTH * g;
  write(G_W, NW, u, out, 0);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// Do the ending fft_WIDTH after an fftMiddleOut.  This is the same as the first half of carryFused.
KERNEL(G_W) fftW(P(T2) out, CP(T2) in, Trig smallTrig) {
  local F2 lds[WIDTH / 2];

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;

  F2 u[NW];
  u32 g = get_group_id(0);

  readCarryFusedLine(inF2, u, g);
  fft_WIDTH(lds, u, smallTrigF2);  
  outF2 += WIDTH * g;
  write(G_W, NW, u, outF2, 0);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

KERNEL(G_W) fftWGF31(P(T2) out, CP(T2) in, Trig smallTrig) {
  local GF31 lds[WIDTH / 2];

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);

  GF31 u[NW];
  u32 g = get_group_id(0);

  readCarryFusedLine(in31, u, g);
  fft_WIDTH(lds, u, smallTrig31);
  out31 += WIDTH * g;
  write(G_W, NW, u, out31, 0);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

KERNEL(G_W) fftWGF61(P(T2) out, CP(T2) in, Trig smallTrig) {
  local GF61 lds[WIDTH / 2];

  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTWTRIGGF61);

  GF61 u[NW];
  u32 g = get_group_id(0);

  readCarryFusedLine(in61, u, g);
  fft_WIDTH(lds, u, smallTrig61);  
  out61 += WIDTH * g;
  write(G_W, NW, u, out61, 0);
}

#endif

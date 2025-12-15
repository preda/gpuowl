// Copyright (C) Mihai Preda and George Woltman

#include "base.cl"
#include "tailutil.cl"
#include "trig.cl"
#include "fftheight.cl"

#if FFT_FP64

// Handle the final multiplication step on a pair of complex numbers.  Swap real and imaginary results for the inverse FFT.
// We used to conjugate the results, but swapping real and imaginary can save some negations in carry propagation.

void OVERLOAD onePairMul(T2* pa, T2* pb, T2* pc, T2* pd, T2 t_squared) {
  T2 a = *pa, b = *pb, c = *pc, d = *pd;

  X2conjb(a, b);
  X2conjb(c, d);

  *pa = cfma(a, c, cmul(cmul(b, d), -t_squared));
  *pb = cfma(b, c, cmul(a, d));

  X2_conjb(*pa, *pb);

  *pa = SWAP_XY(*pa), *pb = SWAP_XY(*pb);
}

void OVERLOAD pairMul(u32 N, T2 *u, T2 *v, T2 *p, T2 *q, T2 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = SWAP_XY(2 * foo2(u[i], p[i]));
      v[i] = SWAP_XY(4 * cmul(v[i], q[i]));
    } else {
      onePairMul(&u[i], &v[i], &p[i], &q[i], base_squared);
    }

    if (N == NH) {
      onePairMul(&u[i+NH/2], &v[i+NH/2], &p[i+NH/2], &q[i+NH/2], -base_squared);
    }

    T2 new_base_squared = mul_t4(base_squared);
    onePairMul(&u[i+NH/4], &v[i+NH/4], &p[i+NH/4], &q[i+NH/4], new_base_squared);

    if (N == NH) {
      onePairMul(&u[i+3*NH/4], &v[i+3*NH/4], &p[i+3*NH/4], &q[i+3*NH/4], -new_base_squared);
    }
  }
}

KERNEL(G_H) tailMul(P(T2) out, CP(T2) in, CP(T2) a, Trig smallTrig) {
  local T2 lds[SMALL_HEIGHT];

  T2 u[NH], v[NH];
  T2 p[NH], q[NH];

  u32 H = ND / SMALL_HEIGHT;

  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);

  u32 me = get_local_id(0);
  readTailFusedLine(in, u, line1, me);
  readTailFusedLine(in, v, line2, me);

#if NH == 8
  T2 w = fancyTrig_N(ND / SMALL_HEIGHT * me);
#else
  T2 w = slowTrig_N(ND / SMALL_HEIGHT * me, ND / NH);
#endif

#if MUL_LOW
  read(G_H, NH, p, a, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, a, memline2 * SMALL_HEIGHT);
  fft_HEIGHT(lds, u, smallTrig, w);
  bar();
  fft_HEIGHT(lds, v, smallTrig, w);
#else
  readTailFusedLine(a, p, line1, me);
  readTailFusedLine(a, q, line2, me);
  fft_HEIGHT(lds, u, smallTrig, w);
  bar();
  fft_HEIGHT(lds, v, smallTrig, w);
  bar();
  fft_HEIGHT(lds, p, smallTrig, w);
  bar();
  fft_HEIGHT(lds, q, smallTrig, w);
#endif

  T2 trig = slowTrig_N(line1 + me * H, ND / NH);

  if (line1 == 0) {
    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, trig, true);
    reverse(G_H, lds, u + NH/2, true);

    T2 trig2 = cmulFancy(trig, TAILT);
    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, trig2, false);
    reverse(G_H, lds, v + NH/2, false);
  } else {
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);
    pairMul(NH, u, v, p, q, trig, false);
    reverseLine(G_H, lds, v);
  }

  bar();
  fft_HEIGHT(lds, v, smallTrig, w);
  bar();
  fft_HEIGHT(lds, u, smallTrig, w);
  writeTailFusedLine(v, out, memline2, me);
  writeTailFusedLine(u, out, memline1, me);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// Handle the final multiplication step on a pair of complex numbers.  Swap real and imaginary results for the inverse FFT.
// We used to conjugate the results, but swapping real and imaginary can save some negations in carry propagation.

void OVERLOAD onePairMul(F2* pa, F2* pb, F2* pc, F2* pd, F2 t_squared) {
  F2 a = *pa, b = *pb, c = *pc, d = *pd;
  X2conjb(a, b);
  X2conjb(c, d);
  *pa = cfma(a, c, cmul(cmul(b, d), -t_squared));
  *pb = cfma(b, c, cmul(a, d));
  X2_conjb(*pa, *pb);
  *pa = SWAP_XY(*pa), *pb = SWAP_XY(*pb);
}

void OVERLOAD pairMul(u32 N, F2 *u, F2 *v, F2 *p, F2 *q, F2 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = SWAP_XY(2 * foo2(u[i], p[i]));
      v[i] = SWAP_XY(4 * cmul(v[i], q[i]));
    } else {
      onePairMul(&u[i], &v[i], &p[i], &q[i], base_squared);
    }

    if (N == NH) {
      onePairMul(&u[i+NH/2], &v[i+NH/2], &p[i+NH/2], &q[i+NH/2], -base_squared);
    }

    F2 new_base_squared = mul_t4(base_squared);
    onePairMul(&u[i+NH/4], &v[i+NH/4], &p[i+NH/4], &q[i+NH/4], new_base_squared);

    if (N == NH) {
      onePairMul(&u[i+3*NH/4], &v[i+3*NH/4], &p[i+3*NH/4], &q[i+3*NH/4], -new_base_squared);
    }
  }
}

KERNEL(G_H) tailMul(P(T2) out, CP(T2) in, CP(T2) a, Trig smallTrig) {
  local F2 lds[SMALL_HEIGHT];

  CP(F2) inF2 = (CP(F2)) in;
  CP(F2) aF2 = (CP(F2)) a;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;

  F2 u[NH], v[NH];
  F2 p[NH], q[NH];

  u32 H = ND / SMALL_HEIGHT;

  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);

  u32 me = get_local_id(0);
  readTailFusedLine(inF2, u, line1, me);
  readTailFusedLine(inF2, v, line2, me);

#if MUL_LOW
  read(G_H, NH, p, aF2, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, aF2, memline2 * SMALL_HEIGHT);
  fft_HEIGHT(lds, u, smallTrigF2);
  bar();
  fft_HEIGHT(lds, v, smallTrigF2);
#else
  readTailFusedLine(aF2, p, line1, me);
  readTailFusedLine(aF2, q, line2, me);
  fft_HEIGHT(lds, u, smallTrigF2);
  bar();
  fft_HEIGHT(lds, v, smallTrigF2);
  bar();
  fft_HEIGHT(lds, p, smallTrigF2);
  bar();
  fft_HEIGHT(lds, q, smallTrigF2);
#endif

  F2 trig = slowTrig_N(line1 + me * H, ND / NH);

  if (line1 == 0) {
    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, trig, true);
    reverse(G_H, lds, u + NH/2, true);

    F2 trig2 = cmulFancy(trig, TAILT);
    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, trig2, false);
    reverse(G_H, lds, v + NH/2, false);
  } else {
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);
    pairMul(NH, u, v, p, q, trig, false);
    reverseLine(G_H, lds, v);
  }

  bar();
  fft_HEIGHT(lds, v, smallTrigF2);
  bar();
  fft_HEIGHT(lds, u, smallTrigF2);
  writeTailFusedLine(v, outF2, memline2, me);
  writeTailFusedLine(u, outF2, memline1, me);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

void OVERLOAD onePairMul(GF31* pa, GF31* pb, GF31* pc, GF31* pd, GF31 t_squared) {
  GF31 a = *pa, b = *pb, c = *pc, d = *pd;

  X2conjb(a, b);
  X2conjb(c, d);

  *pa = sub(cmul(a, c), cmul(cmul(b, d), t_squared));
  *pb = add(cmul(b, c), cmul(a, d));

  X2_conjb(*pa, *pb);
  *pa = SWAP_XY(*pa), *pb = SWAP_XY(*pb);
}

void OVERLOAD pairMul(u32 N, GF31 *u, GF31 *v, GF31 *p, GF31 *q, GF31 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = SWAP_XY(mul2(foo2(u[i], p[i])));
      v[i] = SWAP_XY(shl(cmul(v[i], q[i]), 2));
   } else {
      onePairMul(&u[i], &v[i], &p[i], &q[i], base_squared);
    }

    if (N == NH) {
      onePairMul(&u[i+NH/2], &v[i+NH/2], &p[i+NH/2], &q[i+NH/2], neg(base_squared));
    }

    GF31 new_base_squared = mul_t4(base_squared);
    onePairMul(&u[i+NH/4], &v[i+NH/4], &p[i+NH/4], &q[i+NH/4], new_base_squared);

    if (N == NH) {
      onePairMul(&u[i+3*NH/4], &v[i+3*NH/4], &p[i+3*NH/4], &q[i+3*NH/4], neg(new_base_squared));
    }
  }
}

KERNEL(G_H) tailMulGF31(P(T2) out, CP(T2) in, CP(T2) a, Trig smallTrig) {
  local GF31 lds[SMALL_HEIGHT];

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  CP(GF31) a31 = (CP(GF31)) (a + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTHTRIGGF31);

  GF31 u[NH], v[NH];
  GF31 p[NH], q[NH];

  u32 H = ND / SMALL_HEIGHT;

  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);

  u32 me = get_local_id(0);
  readTailFusedLine(in31, u, line1, me);
  readTailFusedLine(in31, v, line2, me);

#if MUL_LOW
  read(G_H, NH, p, a31, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, a31, memline2 * SMALL_HEIGHT);
  fft_HEIGHT(lds, u, smallTrig31);
  bar();
  fft_HEIGHT(lds, v, smallTrig31);
#else
  readTailFusedLine(a31, p, line1, me);
  readTailFusedLine(a31, q, line2, me);
  fft_HEIGHT(lds, u, smallTrig31);
  bar();
  fft_HEIGHT(lds, v, smallTrig31);
  bar();
  fft_HEIGHT(lds, p, smallTrig31);
  bar();
  fft_HEIGHT(lds, q, smallTrig31);
#endif

  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
#if TAIL_TRIGS31 >= 1
  GF31 trig = smallTrig31[height_trigs + me];                    // Trig values for line zero, should be cached
#if SINGLE_WIDE
  GF31 mult = smallTrig31[height_trigs + G_H + line1];
#else
  GF31 mult = smallTrig31[height_trigs + G_H + line1 * 2];
#endif
  trig = cmul(trig, mult);
#else
#if SINGLE_WIDE
  GF31 trig = NTLOAD(smallTrig31[height_trigs + line1*G_H + me]);
#else
  GF31 trig = NTLOAD(smallTrig31[height_trigs + line1*2*G_H + me]);
#endif
#endif

  if (line1 == 0) {
    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, trig, true);
    reverse(G_H, lds, u + NH/2, true);

    GF31 trig2 = cmul(trig, TAILTGF31);
    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, trig2, false);
    reverse(G_H, lds, v + NH/2, false);
  } else {
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);
    pairMul(NH, u, v, p, q, trig, false);
    reverseLine(G_H, lds, v);
  }

  bar();
  fft_HEIGHT(lds, v, smallTrig31);
  bar();
  fft_HEIGHT(lds, u, smallTrig31);
  writeTailFusedLine(v, out31, memline2, me);
  writeTailFusedLine(u, out31, memline1, me);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

void OVERLOAD onePairMul(GF61* pa, GF61* pb, GF61* pc, GF61* pd, GF61 t_squared) {
  GF61 a = *pa, b = *pb, c = *pc, d = *pd;

  X2conjb(a, b);
  X2conjb(c, d);
  GF61 e = subq(cmul(a, c), cmul(cmul(b, d), t_squared), 2);    // Max value is 3*M61+epsilon
  GF61 f = addq(cmul(b, c), cmul(a, d));                        // Max value is 2*M61+epsilon
  X2s_conjb(&e, &f, 4, 3);
  *pa = SWAP_XY(e), *pb = SWAP_XY(f);
}

void OVERLOAD pairMul(u32 N, GF61 *u, GF61 *v, GF61 *p, GF61 *q, GF61 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = SWAP_XY(mul2(foo2(u[i], p[i])));
      v[i] = SWAP_XY(shl(cmul(v[i], q[i]), 2));
   } else {
      onePairMul(&u[i], &v[i], &p[i], &q[i], base_squared);
    }

    if (N == NH) {
      onePairMul(&u[i+NH/2], &v[i+NH/2], &p[i+NH/2], &q[i+NH/2], neg(base_squared));
    }

    GF61 new_base_squared = mul_t4(base_squared);
    onePairMul(&u[i+NH/4], &v[i+NH/4], &p[i+NH/4], &q[i+NH/4], new_base_squared);

    if (N == NH) {
      onePairMul(&u[i+3*NH/4], &v[i+3*NH/4], &p[i+3*NH/4], &q[i+3*NH/4], neg(new_base_squared));
    }
  }
}

KERNEL(G_H) tailMulGF61(P(T2) out, CP(T2) in, CP(T2) a, Trig smallTrig) {
  local GF61 lds[SMALL_HEIGHT];

  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  CP(GF61) a61 = (CP(GF61)) (a + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTHTRIGGF61);

  GF61 u[NH], v[NH];
  GF61 p[NH], q[NH];

  u32 H = ND / SMALL_HEIGHT;

  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);

  u32 me = get_local_id(0);
  readTailFusedLine(in61, u, line1, me);
  readTailFusedLine(in61, v, line2, me);

#if MUL_LOW
  read(G_H, NH, p, a61, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, a61, memline2 * SMALL_HEIGHT);
  fft_HEIGHT(lds, u, smallTrig61);
  bar();
  fft_HEIGHT(lds, v, smallTrig61);
#else
  readTailFusedLine(a61, p, line1, me);
  readTailFusedLine(a61, q, line2, me);
  fft_HEIGHT(lds, u, smallTrig61);
  bar();
  fft_HEIGHT(lds, v, smallTrig61);
  bar();
  fft_HEIGHT(lds, p, smallTrig61);
  bar();
  fft_HEIGHT(lds, q, smallTrig61);
#endif

  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
#if TAIL_TRIGS61 >= 1
  GF61 trig = smallTrig61[height_trigs + me];                    // Trig values for line zero, should be cached
#if SINGLE_WIDE
  GF61 mult = smallTrig61[height_trigs + G_H + line1];
#else
  GF61 mult = smallTrig61[height_trigs + G_H + line1 * 2];
#endif
  trig = cmul(trig, mult);
#else
#if SINGLE_WIDE
  GF61 trig = NTLOAD(smallTrig61[height_trigs + line1*G_H + me]);
#else
  GF61 trig = NTLOAD(smallTrig61[height_trigs + line1*2*G_H + me]);
#endif
#endif

  if (line1 == 0) {
    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, trig, true);
    reverse(G_H, lds, u + NH/2, true);

    GF61 trig2 = cmul(trig, TAILTGF61);
    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, trig2, false);
    reverse(G_H, lds, v + NH/2, false);
  } else {
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);
    pairMul(NH, u, v, p, q, trig, false);
    reverseLine(G_H, lds, v);
  }

  bar();
  fft_HEIGHT(lds, v, smallTrig61);
  bar();
  fft_HEIGHT(lds, u, smallTrig61);
  writeTailFusedLine(v, out61, memline2, me);
  writeTailFusedLine(u, out61, memline1, me);
}

#endif

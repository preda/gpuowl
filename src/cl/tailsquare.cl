// Copyright (C) Mihai Preda and George Woltman

#include "tailutil.cl"
#include "trig.cl"
#include "fftheight.cl"

#if FFT_FP64

// Handle the final squaring step on a pair of complex numbers.  Swap real and imaginary results for the inverse FFT.
// We used to conjugate the results, but swapping real and imaginary can save some negations in carry propagation.
void OVERLOAD onePairSq(T2* pa, T2* pb, T2 t_squared) {
  T2 a = *pa;
  T2 b = *pb;

//  X2conjb(a, b);
//  *pb = mul2(cmul(a, b));
//  *pa = csqa(a, cmul(csq(b), -t_squared));
//  X2_conjb(*pa, *pb);
//  *pa = SWAP_XY(*pa), *pb = SWAP_XY(*pb)

  // Less readable version of the above that saves one complex add by using FMA instructions
  X2conjb(a, b);
  T2 twoab = mul2(cmul(a, b));                          // 2ab
  *pa = csqa(a, cfma(csq(b), -t_squared, twoab));       // final a = a^2 + 2ab - (bt)^2
  (*pb).x = fma(-2.0, twoab.x, (*pa).x);                // final b = a^2 - 2ab - (bt)^2
  (*pb).y = fma(2.0, twoab.y, -(*pa).y);                // conjugate(final b)
  *pa = SWAP_XY(*pa), *pb = SWAP_XY(*pb);
}

void OVERLOAD pairSq(u32 N, T2 *u, T2 *v, T2 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = SWAP_XY(2 * foo(u[i]));
      v[i] = SWAP_XY(4 * csq(v[i]));
    } else {
      onePairSq(&u[i], &v[i], base_squared);
    }

    if (N == NH) {
      onePairSq(&u[i+NH/2], &v[i+NH/2], -base_squared);
    }

    T2 new_base_squared = mul_t4(base_squared);
    onePairSq(&u[i+NH/4], &v[i+NH/4], new_base_squared);

    if (N == NH) {
      onePairSq(&u[i+3*NH/4], &v[i+3*NH/4], -new_base_squared);
    }
  }
}

// The kernel tailSquareZero handles the special cases in tailSquare, i.e. the lines 0 and H/2
// This kernel is launched with 2 workgroups (handling line 0, resp. H/2)
KERNEL(G_H) tailSquareZero(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[SMALL_HEIGHT / 2];
  T2 u[NH];
  u32 H = ND / SMALL_HEIGHT;

  // This kernel in executed in two workgroups.
  u32 which = get_group_id(0);
  assert(which < 2);

  u32 line = which ? (H/2) : 0;
  u32 me = get_local_id(0);
  readTailFusedLine(in, u, line, me);

#if NH == 8
  T2 w = fancyTrig_N(ND / SMALL_HEIGHT * me);
#else
  T2 w = slowTrig_N(ND / SMALL_HEIGHT * me, ND / NH);
#endif

  T2 trig = slowTrig_N(line + me * H, ND / NH);

  fft_HEIGHT(lds, u, smallTrig, w);
  reverse(G_H, lds, u + NH/2, !which);
  pairSq(NH/2, u,   u + NH/2, trig, !which);
  reverse(G_H, lds, u + NH/2, !which);

  bar();
  fft_HEIGHT(lds, u, smallTrig, w);
  writeTailFusedLine(u, out, transPos(line, MIDDLE, WIDTH), me);
}

#if SINGLE_WIDE

KERNEL(G_H) tailSquare(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[SMALL_HEIGHT];

  T2 u[NH], v[NH];

  u32 H = ND / SMALL_HEIGHT;

#if SINGLE_KERNEL
  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);
#else
  u32 line1 = get_group_id(0) + 1;
  u32 line2 = H - line1;
#endif
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

#if ZEROHACK_H
  u32 zerohack = (u32) get_group_id(0) / 131072;
  fft_HEIGHT(lds + zerohack, u, smallTrig + zerohack, w);
  bar();
  fft_HEIGHT(lds + zerohack, v, smallTrig + zerohack, w);
#else
  fft_HEIGHT(lds, u, smallTrig, w);
  bar();
  fft_HEIGHT(lds, v, smallTrig, w);
#endif

  // Compute trig values from scratch.  Good on GPUs with high DP throughput.
#if TAIL_TRIGS == 2
  T2 trig = slowTrig_N(line1 + me * H, ND / NH);

  // Do a little bit of memory access and a little bit of DP math.  Good on a Radeon VII.
#elif TAIL_TRIGS == 1
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*5;
  // Read a hopefully cached line of data and one non-cached T2 per line
  T2 trig = smallTrig[height_trigs + me];                    // Trig values for line zero, should be cached
  T2 mult = smallTrig[height_trigs + G_H + line1];           // Line multiplier
  trig = cmulFancy(trig, mult);

  // On consumer-grade GPUs, it is likely beneficial to read all trig values.
#else
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*5;
  // Read pre-computed trig values
  T2 trig = NTLOAD(smallTrig[height_trigs + line1*G_H + me]);
#endif

#if SINGLE_KERNEL
  if (line1 == 0) {
    // Line 0 is special: it pairs with itself, offseted by 1.
    reverse(G_H, lds, u + NH/2, true);
    pairSq(NH/2, u,   u + NH/2, trig, true);
    reverse(G_H, lds, u + NH/2, true);

    // Line H/2 also pairs with itself (but without offset).
    T2 trig2 = cmulFancy(trig, TAILT);
    reverse(G_H, lds, v + NH/2, false);
    pairSq(NH/2, v,   v + NH/2, trig2, false);
    reverse(G_H, lds, v + NH/2, false);
  }
  else {
#else
  if (1) {
#endif
    reverseLine(G_H, lds, v);
    pairSq(NH, u, v, trig, false);
    reverseLine(G_H, lds, v);
  }

  bar();
  fft_HEIGHT(lds, v, smallTrig, w);
  bar();
  fft_HEIGHT(lds, u, smallTrig, w);

  writeTailFusedLine(v, out, memline2, me);
  writeTailFusedLine(u, out, memline1, me);
}


//
// Create a kernel that uses a double-wide workgroup (u in half the workgroup, v in the other half)
// We hope to get better occupancy with the reduced register usage
//

#else

// Special pairSq for double-wide line 0
void OVERLOAD pairSq2_special(T2 *u, T2 base_squared) {
  u32 me = get_local_id(0);
  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (i == 0 && me == 0) {
      u[0] = SWAP_XY(2 * foo(u[0]));
      u[NH/2] = SWAP_XY(4 * csq(u[NH/2]));
    } else {
      onePairSq(&u[i], &u[NH/2+i], base_squared);
    }
    T2 new_base_squared = mul_t4(base_squared);
    onePairSq(&u[i+NH/4], &u[NH/2+i+NH/4], new_base_squared);
  }
}

KERNEL(G_H * 2) tailSquare(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[SMALL_HEIGHT];

  T2 u[NH];

  u32 H = ND / SMALL_HEIGHT;

#if SINGLE_KERNEL
  u32 line_u = get_group_id(0);
  u32 line_v = line_u ? H - line_u : (H / 2);
#else
  u32 line_u = get_group_id(0) + 1;
  u32 line_v = H - line_u;
#endif

  u32 me = get_local_id(0);
  u32 lowMe = me % G_H;  // lane-id in one of the two halves (half-workgroups).

  // We're going to call the halves "first-half" and "second-half".
  bool isSecondHalf = me >= G_H;

  u32 line = !isSecondHalf ? line_u : line_v;

  // Read lines u and v
  readTailFusedLine(in, u, line, lowMe);

#if NH == 8
  T2 w = fancyTrig_N(H * lowMe);
#else
  T2 w = slowTrig_N(H * lowMe, ND / NH);
#endif

#if ZEROHACK_H
  u32 zerohack = (u32) get_group_id(0) / 131072;
  new_fft_HEIGHT2_1(lds + zerohack, u, smallTrig + zerohack, w);
#else
  new_fft_HEIGHT2_1(lds, u, smallTrig, w);
#endif

  // Compute trig values from scratch.  Good on GPUs with high DP throughput.
#if TAIL_TRIGS == 2
  T2 trig = slowTrig_N(line + H * lowMe, ND / NH * 2);

  // Do a little bit of memory access and a little bit of DP math.  Good on a Radeon VII.
#elif TAIL_TRIGS == 1
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*5;
  // Read a hopefully cached line of data and one non-cached T2 per line
  T2 trig = smallTrig[height_trigs + lowMe];                                 // Trig values for line zero, should be cached
  T2 mult = smallTrig[height_trigs + G_H + line_u*2 + isSecondHalf];         // Two multipliers.  One for line u, one for line v.
  trig = cmulFancy(trig, mult);

  // On consumer-grade GPUs, it is likely beneficial to read all trig values.
#else
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*5;
  // Read pre-computed trig values
  T2 trig = NTLOAD(smallTrig[height_trigs + line_u*G_H*2 + me]);
#endif

  bar(G_H);

#if SINGLE_KERNEL
  // Line 0 and H/2 are special: they pair with themselves, line 0 is offseted by 1.
  if (line_u == 0) {
    reverse2(lds, u);
    pairSq2_special(u, trig);
    reverse2(lds, u);
  }
  else {
#else
  if (1) {
#endif
    revCrossLine(G_H, lds, u + NH/2, NH/2, isSecondHalf);
    pairSq(NH/2, u, u + NH/2, trig, false);
    bar(G_H);
    revCrossLine(G_H, lds, u + NH/2, NH/2, !isSecondHalf);
  }

  bar(G_H);

  new_fft_HEIGHT2_2(lds, u, smallTrig, w);

  // Write lines u and v
  writeTailFusedLine(u, out, transPos(line, MIDDLE, WIDTH), lowMe);
}

#endif

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

// Handle the final squaring step on a pair of complex numbers.  Swap real and imaginary results for the inverse FFT.
// We used to conjugate the results, but swapping real and imaginary can save some negations in carry propagation.
void OVERLOAD onePairSq(F2* pa, F2* pb, F2 t_squared) {
  F2 a = *pa;
  F2 b = *pb;

  X2conjb(a, b);
  F2 twoab = mul2(cmul(a, b));                          // 2ab
  *pa = csqa(a, cfma(csq(b), -t_squared, twoab));       // final a = a^2 + 2ab - (bt)^2
  (*pb).x = fma(-2.0f, twoab.x, (*pa).x);               // final b = a^2 - 2ab - (bt)^2
  (*pb).y = fma(2.0f, twoab.y, -(*pa).y);               // conjugate(final b)
  *pa = SWAP_XY(*pa), *pb = SWAP_XY(*pb);
}

void OVERLOAD pairSq(u32 N, F2 *u, F2 *v, F2 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = SWAP_XY(2 * foo(u[i]));
      v[i] = SWAP_XY(4 * csq(v[i]));
    } else {
      onePairSq(&u[i], &v[i], base_squared);
    }

    if (N == NH) {
      onePairSq(&u[i+NH/2], &v[i+NH/2], -base_squared);
    }

    F2 new_base_squared = mul_t4(base_squared);
    onePairSq(&u[i+NH/4], &v[i+NH/4], new_base_squared);

    if (N == NH) {
      onePairSq(&u[i+3*NH/4], &v[i+3*NH/4], -new_base_squared);
    }
  }
}

// The kernel tailSquareZero handles the special cases in tailSquare, i.e. the lines 0 and H/2
// This kernel is launched with 2 workgroups (handling line 0, resp. H/2)
KERNEL(G_H) tailSquareZero(P(T2) out, CP(T2) in, Trig smallTrig) {
  local F2 lds[SMALL_HEIGHT / 2];
  F2 u[NH];
  u32 H = ND / SMALL_HEIGHT;

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;

  // This kernel in executed in two workgroups.
  u32 which = get_group_id(0);
  assert(which < 2);

  u32 line = which ? (H/2) : 0;
  u32 me = get_local_id(0);
  readTailFusedLine(inF2, u, line, me);

  F2 trig = slowTrig_N(line + me * H, ND / NH);

  fft_HEIGHT(lds, u, smallTrigF2);
  reverse(G_H, lds, u + NH/2, !which);
  pairSq(NH/2, u,   u + NH/2, trig, !which);
  reverse(G_H, lds, u + NH/2, !which);

  bar();
  fft_HEIGHT(lds, u, smallTrigF2);
  writeTailFusedLine(u, outF2, transPos(line, MIDDLE, WIDTH), me);
}

#if SINGLE_WIDE

KERNEL(G_H) tailSquare(P(T2) out, CP(T2) in, Trig smallTrig) {
  local F2 lds[SMALL_HEIGHT];

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;

  F2 u[NH], v[NH];

  u32 H = ND / SMALL_HEIGHT;

#if SINGLE_KERNEL
  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);
#else
  u32 line1 = get_group_id(0) + 1;
  u32 line2 = H - line1;
#endif
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);

  u32 me = get_local_id(0);
  readTailFusedLine(inF2, u, line1, me);
  readTailFusedLine(inF2, v, line2, me);

#if ZEROHACK_H
  u32 zerohack = get_group_id(0) / 131072;
  fft_HEIGHT(lds + zerohack, u, smallTrigF2 + zerohack);
  bar();
  fft_HEIGHT(lds + zerohack, v, smallTrigF2 + zerohack);
#else
  fft_HEIGHT(lds, u, smallTrigF2);
  bar();
  fft_HEIGHT(lds, v, smallTrigF2);
#endif

  // Compute trig values from scratch.  Good on GPUs with high DP throughput.
#if TAIL_TRIGS32 == 2
  F2 trig = slowTrig_N(line1 + me * H, ND / NH);

  // Do a little bit of memory access and a little bit of DP math.  Good on a Radeon VII.
#elif TAIL_TRIGS32 == 1
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
  // Read a hopefully cached line of data and one non-cached F2 per line
  F2 trig = smallTrigF2[height_trigs + me];                    // Trig values for line zero, should be cached
  F2 mult = smallTrigF2[height_trigs + G_H + line1];           // Line multiplier
  trig = cmulFancy(trig, mult);

  // On consumer-grade GPUs, it is likely beneficial to read all trig values.
#else
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
  // Read pre-computed trig values
  F2 trig = NTLOAD(smallTrigF2[height_trigs + line1*G_H + me]);
#endif

#if SINGLE_KERNEL
  if (line1 == 0) {
    // Line 0 is special: it pairs with itself, offseted by 1.
    reverse(G_H, lds, u + NH/2, true);
    pairSq(NH/2, u,   u + NH/2, trig, true);
    reverse(G_H, lds, u + NH/2, true);

    // Line H/2 also pairs with itself (but without offset).
    F2 trig2 = cmulFancy(trig, TAILT);
    reverse(G_H, lds, v + NH/2, false);
    pairSq(NH/2, v,   v + NH/2, trig2, false);
    reverse(G_H, lds, v + NH/2, false);
  }
  else {
#else
  if (1) {
#endif
    reverseLine(G_H, lds, v);
    pairSq(NH, u, v, trig, false);
    reverseLine(G_H, lds, v);
  }

  bar();
  fft_HEIGHT(lds, v, smallTrigF2);
  bar();
  fft_HEIGHT(lds, u, smallTrigF2);

  writeTailFusedLine(v, outF2, memline2, me);
  writeTailFusedLine(u, outF2, memline1, me);
}


//
// Create a kernel that uses a double-wide workgroup (u in half the workgroup, v in the other half)
// We hope to get better occupancy with the reduced register usage
//

#else

// Special pairSq for double-wide line 0
void OVERLOAD pairSq2_special(F2 *u, F2 base_squared) {
  u32 me = get_local_id(0);
  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (i == 0 && me == 0) {
      u[0] = SWAP_XY(2 * foo(u[0]));
      u[NH/2] = SWAP_XY(4 * csq(u[NH/2]));
    } else {
      onePairSq(&u[i], &u[NH/2+i], base_squared);
    }
    F2 new_base_squared = mul_t4(base_squared);
    onePairSq(&u[i+NH/4], &u[NH/2+i+NH/4], new_base_squared);
  }
}

KERNEL(G_H * 2) tailSquare(P(T2) out, CP(T2) in, Trig smallTrig) {
  local F2 lds[SMALL_HEIGHT];

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;

  F2 u[NH];

  u32 H = ND / SMALL_HEIGHT;

#if SINGLE_KERNEL
  u32 line_u = get_group_id(0);
  u32 line_v = line_u ? H - line_u : (H / 2);
#else
  u32 line_u = get_group_id(0) + 1;
  u32 line_v = H - line_u;
#endif

  u32 me = get_local_id(0);
  u32 lowMe = me % G_H;  // lane-id in one of the two halves (half-workgroups).

  // We're going to call the halves "first-half" and "second-half".
  bool isSecondHalf = me >= G_H;

  u32 line = !isSecondHalf ? line_u : line_v;

  // Read lines u and v
  readTailFusedLine(inF2, u, line, lowMe);

#if ZEROHACK_H
  u32 zerohack = (u32) get_group_id(0) / 131072;
  new_fft_HEIGHT2_1(lds + zerohack, u, smallTrigF2 + zerohack);
#else
  new_fft_HEIGHT2_1(lds, u, smallTrigF2);
#endif

  // Compute trig values from scratch.  Good on GPUs with high DP throughput.
#if TAIL_TRIGS32 == 2
  F2 trig = slowTrig_N(line + H * lowMe, ND / NH * 2);

  // Do a little bit of memory access and a little bit of DP math.  Good on a Radeon VII.
#elif TAIL_TRIGS32 == 1
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
  // Read a hopefully cached line of data and one non-cached F2 per line
  F2 trig = smallTrigF2[height_trigs + lowMe];                                 // Trig values for line zero, should be cached
  F2 mult = smallTrigF2[height_trigs + G_H + line_u*2 + isSecondHalf];         // Two multipliers.  One for line u, one for line v.
  trig = cmulFancy(trig, mult);

  // On consumer-grade GPUs, it is likely beneficial to read all trig values.
#else
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
  // Read pre-computed trig values
  F2 trig = NTLOAD(smallTrigF2[height_trigs + line_u*G_H*2 + me]);
#endif

  bar(G_H);

#if SINGLE_KERNEL
  // Line 0 and H/2 are special: they pair with themselves, line 0 is offseted by 1.
  if (line_u == 0) {
    reverse2(lds, u);
    pairSq2_special(u, trig);
    reverse2(lds, u);
  }
  else {
#else
  if (1) {
#endif
    revCrossLine(G_H, lds, u + NH/2, NH/2, isSecondHalf);
    pairSq(NH/2, u, u + NH/2, trig, false);
    bar(G_H);
    revCrossLine(G_H, lds, u + NH/2, NH/2, !isSecondHalf);
  }

  bar(G_H);

  new_fft_HEIGHT2_2(lds, u, smallTrigF2);

  // Write lines u and v
  writeTailFusedLine(u, outF2, transPos(line, MIDDLE, WIDTH), lowMe);
}

#endif

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

void OVERLOAD onePairSq(GF31* pa, GF31* pb, GF31 t_squared, const u32 t_squared_type) {
  GF31 a = *pa, b = *pb;
  GF31 c, d;

  X2conjb(a, b);
  if (t_squared_type == 0)                           // mul t_squared by 1
    c = csq_sub(a, cmul(csq(b), t_squared));         // a^2 - (b^2 * t_squared)
  if (t_squared_type == 1)                           // mul t_squared by i
    c = csq_subi(a, cmul(csq(b), t_squared));        // a^2 - i*(b^2 * t_squared)
  if (t_squared_type == 2)                           // mul t_squared by -1
    c = csq_add(a, cmul(csq(b), t_squared));         // a^2 - -1*(b^2 * t_squared)
  if (t_squared_type == 3)                           // mul t_squared by -i
    c = csq_addi(a, cmul(csq(b), t_squared));        // a^2 - -i*(b^2 * t_squared)
  d = mul2(cmul(a, b));
  X2_conjb(c, d);
  *pa = SWAP_XY(c), *pb = SWAP_XY(d);
}

void OVERLOAD pairSq(u32 N, GF31 *u, GF31 *v, GF31 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = SWAP_XY(mul2(foo(u[i])));
      v[i] = SWAP_XY(shl(csq(v[i]), 2));
    } else {
      onePairSq(&u[i], &v[i], base_squared, 0);
    }

    if (N == NH) {
      onePairSq(&u[i+NH/2], &v[i+NH/2], base_squared, 2);
    }

    onePairSq(&u[i+NH/4], &v[i+NH/4], base_squared, 1);

    if (N == NH) {
      onePairSq(&u[i+3*NH/4], &v[i+3*NH/4], base_squared, 3);
    }
  }
}

// The kernel tailSquareZero handles the special cases in tailSquare, i.e. the lines 0 and H/2
// This kernel is launched with 2 workgroups (handling line 0, resp. H/2)
KERNEL(G_H) tailSquareZeroGF31(P(T2) out, CP(T2) in, Trig smallTrig) {
  local GF31 lds[SMALL_HEIGHT / 2];

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTHTRIGGF31);

  GF31 u[NH];
  u32 H = ND / SMALL_HEIGHT;

  // This kernel in executed in two workgroups.
  u32 which = get_group_id(0);
  assert(which < 2);

  u32 line = which ? (H/2) : 0;
  u32 me = get_local_id(0);
  readTailFusedLine(in31, u, line, me);

  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
#if TAIL_TRIGS31 >= 1
  GF31 trig = smallTrig31[height_trigs + me];
#if SINGLE_WIDE
  GF31 mult = smallTrig31[height_trigs + G_H + line];
#else
  GF31 mult = smallTrig31[height_trigs + G_H + which];
#endif
  trig = cmul(trig, mult);
#else
#if SINGLE_WIDE
  GF31 trig = NTLOAD(smallTrig31[height_trigs + line*G_H + me]);
#else
  GF31 trig = NTLOAD(smallTrig31[height_trigs + which*G_H + me]);
#endif
#endif

  fft_HEIGHT(lds, u, smallTrig31);
  reverse(G_H, lds, u + NH/2, !which);
  pairSq(NH/2, u,   u + NH/2, trig, !which);
  reverse(G_H, lds, u + NH/2, !which);
  bar();
  fft_HEIGHT(lds, u, smallTrig31);
  writeTailFusedLine(u, out31, transPos(line, MIDDLE, WIDTH), me);
}

#if SINGLE_WIDE

KERNEL(G_H) tailSquareGF31(P(T2) out, CP(T2) in, Trig smallTrig) {
  local GF31 lds[SMALL_HEIGHT];

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTHTRIGGF31);

  GF31 u[NH], v[NH];

  u32 H = ND / SMALL_HEIGHT;

#if SINGLE_KERNEL
  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);
#else
  u32 line1 = get_group_id(0) + 1;
  u32 line2 = H - line1;
#endif
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);

  u32 me = get_local_id(0);
  readTailFusedLine(in31, u, line1, me);
  readTailFusedLine(in31, v, line2, me);

#if ZEROHACK_H
  u32 zerohack = (u32) get_group_id(0) / 131072;
  fft_HEIGHT(lds + zerohack, u, smallTrig31 + zerohack);
  bar();
  fft_HEIGHT(lds + zerohack, v, smallTrig31 + zerohack);
#else
  fft_HEIGHT(lds, u, smallTrig31);
  bar();
  fft_HEIGHT(lds, v, smallTrig31);
#endif

  // Do a little bit of memory access and a little bit of math.
#if TAIL_TRIGS31 >= 1
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
  // Read a hopefully cached line of data and one non-cached GF31 per line
  GF31 trig = smallTrig31[height_trigs + me];                    // Trig values for line zero, should be cached
  GF31 mult = smallTrig31[height_trigs + G_H + line1];           // Line multiplier
  trig = cmul(trig, mult);

  // On consumer-grade GPUs, it is likely beneficial to read all trig values.
#else
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
  // Read pre-computed trig values
  GF31 trig = NTLOAD(smallTrig31[height_trigs + line1*G_H + me]);
#endif

#if SINGLE_KERNEL
  if (line1 == 0) {
    // Line 0 is special: it pairs with itself, offseted by 1.
    reverse(G_H, lds, u + NH/2, true);
    pairSq(NH/2, u,   u + NH/2, trig, true);
    reverse(G_H, lds, u + NH/2, true);

    // Line H/2 also pairs with itself (but without offset).
    GF31 trig2 = cmul(trig, TAILTGF31);
    reverse(G_H, lds, v + NH/2, false);
    pairSq(NH/2, v,   v + NH/2, trig2, false);
    reverse(G_H, lds, v + NH/2, false);
  }
  else {
#else
  if (1) {
#endif
    reverseLine(G_H, lds, v);
    pairSq(NH, u, v, trig, false);
    reverseLine(G_H, lds, v);
  }

  bar();
  fft_HEIGHT(lds, v, smallTrig31);
  bar();
  fft_HEIGHT(lds, u, smallTrig31);

  writeTailFusedLine(v, out31, memline2, me);
  writeTailFusedLine(u, out31, memline1, me);
}


//
// Create a kernel that uses a double-wide workgroup (u in half the workgroup, v in the other half)
// We hope to get better occupancy with the reduced register usage
//

#else

// Special pairSq for double-wide line 0
void OVERLOAD pairSq2_special(GF31 *u, GF31 base_squared) {
  u32 me = get_local_id(0);
  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (i == 0 && me == 0) {
      u[0] = SWAP_XY(mul2(foo(u[0])));
      u[NH/2] = SWAP_XY(shl(csq(u[NH/2]), 2));
    } else {
      onePairSq(&u[i], &u[NH/2+i], base_squared, 0);
    }
    onePairSq(&u[i+NH/4], &u[NH/2+i+NH/4], base_squared, 1);
  }
}

KERNEL(G_H * 2) tailSquareGF31(P(T2) out, CP(T2) in, Trig smallTrig) {
  local GF31 lds[SMALL_HEIGHT];

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTHTRIGGF31);

  GF31 u[NH];

  u32 H = ND / SMALL_HEIGHT;

#if SINGLE_KERNEL
  u32 line_u = get_group_id(0);
  u32 line_v = line_u ? H - line_u : (H / 2);
#else
  u32 line_u = get_group_id(0) + 1;
  u32 line_v = H - line_u;
#endif

  u32 me = get_local_id(0);
  u32 lowMe = me % G_H;  // lane-id in one of the two halves (half-workgroups).

  // We're going to call the halves "first-half" and "second-half".
  bool isSecondHalf = me >= G_H;

  u32 line = !isSecondHalf ? line_u : line_v;

  // Read lines u and v
  readTailFusedLine(in31, u, line, lowMe);

#if ZEROHACK_H
  u32 zerohack = (u32) get_group_id(0) / 131072;
  new_fft_HEIGHT2_1(lds + zerohack, u, smallTrig31 + zerohack);
#else
  new_fft_HEIGHT2_1(lds, u, smallTrig31);
#endif

  // Do a little bit of memory access and a little bit of math.  Good on a Radeon VII.
#if TAIL_TRIGS31 >= 1
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
  // Read a hopefully cached line of data and one non-cached GF31 per line
  GF31 trig = smallTrig31[height_trigs + lowMe];                                 // Trig values for line zero, should be cached
  GF31 mult = smallTrig31[height_trigs + G_H + line_u*2 + isSecondHalf];         // Two multipliers.  One for line u, one for line v.
  trig = cmul(trig, mult);

  // On consumer-grade GPUs, it is likely beneficial to read all trig values.
#else
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
  // Read pre-computed trig values
  GF31 trig = NTLOAD(smallTrig31[height_trigs + line_u*G_H*2 + me]);
#endif

  bar(G_H);

#if SINGLE_KERNEL
  // Line 0 and H/2 are special: they pair with themselves, line 0 is offseted by 1.
  if (line_u == 0) {
    reverse2(lds, u);
    pairSq2_special(u, trig);
    reverse2(lds, u);
  }
  else {
#else
  if (1) {
#endif
    revCrossLine(G_H, lds, u + NH/2, NH/2, isSecondHalf);
    pairSq(NH/2, u, u + NH/2, trig, false);
    bar(G_H);
    revCrossLine(G_H, lds, u + NH/2, NH/2, !isSecondHalf);
  }

  bar(G_H);

  new_fft_HEIGHT2_2(lds, u, smallTrig31);

  // Write lines u and v
  writeTailFusedLine(u, out31, transPos(line, MIDDLE, WIDTH), lowMe);
}

#endif

#endif
  

/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

void OVERLOAD onePairSq(GF61* pa, GF61* pb, GF61 t_squared, const u32 t_squared_type) {
  GF61 a = *pa, b = *pb;
  GF61 c, d;

  X2conjb(a, b);
  if (t_squared_type == 0)                             // mul t_squared by 1
    c = subq(csqq(a, 2), cmul(csq(b), t_squared), 2);  // max c value is 4*M61+epsilon
  if (t_squared_type == 1)                             // mul t_squared by i
    c = subiq(csqq(a, 2), cmul(csq(b), t_squared), 2); // max c value is 4*M61+epsilon
  if (t_squared_type == 2)                             // mul t_squared by -1
    c = addq(csqq(a, 2), cmul(csq(b), t_squared));     // max c value is 3*M61+epsilon
  if (t_squared_type == 3)                             // mul t_squared by -i
    c = addiq(csqq(a, 2), cmul(csq(b), t_squared), 2); // max c value is 4*M61+epsilon
  d = 2 * cmul(a, b);                                  // max d value is 2*M61+epsilon
  X2s_conjb(&c, &d, 5, 3);
  *pa = SWAP_XY(c), *pb = SWAP_XY(d);
}

void OVERLOAD pairSq(u32 N, GF61 *u, GF61 *v, GF61 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = SWAP_XY(mul2(foo(u[i])));
      v[i] = SWAP_XY(shl(csq(v[i]), 2));
    } else {
      onePairSq(&u[i], &v[i], base_squared, 0);
    }

    if (N == NH) {
      onePairSq(&u[i+NH/2], &v[i+NH/2], base_squared, 2);
    }

    onePairSq(&u[i+NH/4], &v[i+NH/4], base_squared, 1);

    if (N == NH) {
      onePairSq(&u[i+3*NH/4], &v[i+3*NH/4], base_squared, 3);
    }
  }
}

// The kernel tailSquareZero handles the special cases in tailSquare, i.e. the lines 0 and H/2
// This kernel is launched with 2 workgroups (handling line 0, resp. H/2)
KERNEL(G_H) tailSquareZeroGF61(P(T2) out, CP(T2) in, Trig smallTrig) {
  local GF61 lds[SMALL_HEIGHT / 2];

  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTHTRIGGF61);

  GF61 u[NH];
  u32 H = ND / SMALL_HEIGHT;

  // This kernel in executed in two workgroups.
  u32 which = get_group_id(0);
  assert(which < 2);

  u32 line = which ? (H/2) : 0;
  u32 me = get_local_id(0);
  readTailFusedLine(in61, u, line, me);

  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
#if TAIL_TRIGS61 >= 1
  GF61 trig = smallTrig61[height_trigs + me];
#if SINGLE_WIDE
  GF61 mult = smallTrig61[height_trigs + G_H + line];
#else
  GF61 mult = smallTrig61[height_trigs + G_H + which];
#endif
  trig = cmul(trig, mult);
#else
#if SINGLE_WIDE
  GF61 trig = NTLOAD(smallTrig61[height_trigs + line*G_H + me]);
#else
  GF61 trig = NTLOAD(smallTrig61[height_trigs + which*G_H + me]);
#endif
#endif

  fft_HEIGHT(lds, u, smallTrig61);
  reverse(G_H, lds, u + NH/2, !which);
  pairSq(NH/2, u,   u + NH/2, trig, !which);
  reverse(G_H, lds, u + NH/2, !which);
  bar();
  fft_HEIGHT(lds, u, smallTrig61);
  writeTailFusedLine(u, out61, transPos(line, MIDDLE, WIDTH), me);
}

#if SINGLE_WIDE

KERNEL(G_H) tailSquareGF61(P(T2) out, CP(T2) in, Trig smallTrig) {
  local GF61 lds[SMALL_HEIGHT];

  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTHTRIGGF61);

  GF61 u[NH], v[NH];

  u32 H = ND / SMALL_HEIGHT;

#if SINGLE_KERNEL
  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);
#else
  u32 line1 = get_group_id(0) + 1;
  u32 line2 = H - line1;
#endif
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);

  u32 me = get_local_id(0);
  readTailFusedLine(in61, u, line1, me);
  readTailFusedLine(in61, v, line2, me);

#if ZEROHACK_H
  u32 zerohack = (u32) get_group_id(0) / 131072;
  fft_HEIGHT(lds + zerohack, u, smallTrig61 + zerohack);
  bar();
  fft_HEIGHT(lds + zerohack, v, smallTrig61 + zerohack);
#else
  fft_HEIGHT(lds, u, smallTrig61);
  bar();
  fft_HEIGHT(lds, v, smallTrig61);
#endif

  // Do a little bit of memory access and a little bit of math.
#if TAIL_TRIGS61 >= 1
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
  // Read a hopefully cached line of data and one non-cached GF61 per line
  GF61 trig = smallTrig61[height_trigs + me];                    // Trig values for line zero, should be cached
  GF61 mult = smallTrig61[height_trigs + G_H + line1];           // Line multiplier
  trig = cmul(trig, mult);

  // On consumer-grade GPUs, it is likely beneficial to read all trig values.
#else
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
  // Read pre-computed trig values
  GF61 trig = NTLOAD(smallTrig61[height_trigs + line1*G_H + me]);
#endif

#if SINGLE_KERNEL
  if (line1 == 0) {
    // Line 0 is special: it pairs with itself, offseted by 1.
    reverse(G_H, lds, u + NH/2, true);
    pairSq(NH/2, u,   u + NH/2, trig, true);
    reverse(G_H, lds, u + NH/2, true);

    // Line H/2 also pairs with itself (but without offset).
    GF61 trig2 = cmul(trig, TAILTGF61);
    reverse(G_H, lds, v + NH/2, false);
    pairSq(NH/2, v,   v + NH/2, trig2, false);
    reverse(G_H, lds, v + NH/2, false);
  }
  else {
#else
  if (1) {
#endif
    reverseLine(G_H, lds, v);
    pairSq(NH, u, v, trig, false);
    reverseLine(G_H, lds, v);
  }

  bar();
  fft_HEIGHT(lds, v, smallTrig61);
  bar();
  fft_HEIGHT(lds, u, smallTrig61);

  writeTailFusedLine(v, out61, memline2, me);
  writeTailFusedLine(u, out61, memline1, me);
}


//
// Create a kernel that uses a double-wide workgroup (u in half the workgroup, v in the other half)
// We hope to get better occupancy with the reduced register usage
//

#else

// Special pairSq for double-wide line 0
void OVERLOAD pairSq2_special(GF61 *u, GF61 base_squared) {
  u32 me = get_local_id(0);
  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (i == 0 && me == 0) {
      u[0] = SWAP_XY(mul2(foo(u[0])));
      u[NH/2] = SWAP_XY(shl(csq(u[NH/2]), 2));
    } else {
      onePairSq(&u[i], &u[NH/2+i], base_squared, 0);
    }
    onePairSq(&u[i+NH/4], &u[NH/2+i+NH/4], base_squared, 1);
  }
}

KERNEL(G_H * 2) tailSquareGF61(P(T2) out, CP(T2) in, Trig smallTrig) {
  local GF61 lds[SMALL_HEIGHT];

  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTHTRIGGF61);

  GF61 u[NH];

  u32 H = ND / SMALL_HEIGHT;

#if SINGLE_KERNEL
  u32 line_u = get_group_id(0);
  u32 line_v = line_u ? H - line_u : (H / 2);
#else
  u32 line_u = get_group_id(0) + 1;
  u32 line_v = H - line_u;
#endif

  u32 me = get_local_id(0);
  u32 lowMe = me % G_H;  // lane-id in one of the two halves (half-workgroups).

  // We're going to call the halves "first-half" and "second-half".
  bool isSecondHalf = me >= G_H;

  u32 line = !isSecondHalf ? line_u : line_v;

  // Read lines u and v
  readTailFusedLine(in61, u, line, lowMe);

#if ZEROHACK_H
  u32 zerohack = (u32) get_group_id(0) / 131072;
  new_fft_HEIGHT2_1(lds + zerohack, u, smallTrig61 + zerohack);
#else
  new_fft_HEIGHT2_1(lds, u, smallTrig61);
#endif

  // Do a little bit of memory access and a little bit of math.  Good on a Radeon VII.
#if TAIL_TRIGS61 >= 1
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
  // Read a hopefully cached line of data and one non-cached GF61 per line
  GF61 trig = smallTrig61[height_trigs + lowMe];                                 // Trig values for line zero, should be cached
  GF61 mult = smallTrig61[height_trigs + G_H + line_u*2 + isSecondHalf];         // Two multipliers.  One for line u, one for line v.
  trig = cmul(trig, mult);

  // On consumer-grade GPUs, it is likely beneficial to read all trig values.
#else
  // Calculate number of trig values used by fft_HEIGHT (see genSmallTrigCombo in trigBufCache.cpp)
  // The trig values used here are pre-computed and stored after the fft_HEIGHT trig values.
  u32 height_trigs = SMALL_HEIGHT*1;
  // Read pre-computed trig values
  GF61 trig = NTLOAD(smallTrig61[height_trigs + line_u*G_H*2 + me]);
#endif

  bar(G_H);

#if SINGLE_KERNEL
  // Line 0 and H/2 are special: they pair with themselves, line 0 is offseted by 1.
  if (line_u == 0) {
    reverse2(lds, u);
    pairSq2_special(u, trig);
    reverse2(lds, u);
  }
  else {
#else
  if (1) {
#endif
    revCrossLine(G_H, lds, u + NH/2, NH/2, isSecondHalf);
    pairSq(NH/2, u, u + NH/2, trig, false);
    bar(G_H);
    revCrossLine(G_H, lds, u + NH/2, NH/2, !isSecondHalf);
  }

  bar(G_H);

  new_fft_HEIGHT2_2(lds, u, smallTrig61);

  // Write lines u and v
  writeTailFusedLine(u, out61, transPos(line, MIDDLE, WIDTH), lowMe);
}

#endif

#endif

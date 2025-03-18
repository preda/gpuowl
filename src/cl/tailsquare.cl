// Copyright (C) Mihai Preda and George Woltman

#include "tailutil.cl"
#include "trig.cl"
#include "fftheight.cl"

//#define PREFER_DP_TO_MEM        2       // Excellent DP GPU such as Titan V or Radeon VII Pro.
#define PREFER_DP_TO_MEM      1       // Good DP GPU.  Tuned for Radeon VII.
//#define PREFER_DP_TO_MEM      0       // Poor DP GPU.  A typical consumer grade GPU.

// TAIL_KERNELS setting:
//      0 = single wide, single kernel
//      1 = single wide, two kernels
//      2 = double wide, single kernel
//      3 = double wide, two kernels
#if !defined(TAIL_KERNELS)
#define TAIL_KERNELS    3                         // Default is double-wide tailSquare with two kernels
#endif
#define SINGLE_WIDE     TAIL_KERNELS < 2          // Old single-wide tailSquare vs. new double-wide tailSquare
#define SINGLE_KERNEL   (TAIL_KERNELS & 1) == 0   // TailSquare uses a single kernel vs. two kernels

// Why does this alternate implementation work?  Let t' be the conjugate of t and note that t*t' = 1.
// Now consider these lines from the original implementation (comments appear alongside):
//      b = mul_by_conjugate(b, t);                     bt'
//      X2(a, b);                                       a + bt', a - bt'
//      a = sq(a);                                      a^2 + 2abt' + (bt')^2
//      b = sq(b);                                      a^2 - 2abt' + (bt')^2
//      X2(a, b);                                       2a^2 + 2(bt')^2, 4abt'
//      b = mul(b, t);                                                   4ab

void onePairSq(T2* pa, T2* pb, T2 conjugate_t_squared) {
  T2 a = *pa;
  T2 b = *pb;

//  X2conjb(a, b);
//  *pb = mul2(cmul(a, b));
//  *pa = csqa(a, cmul(csq(b), conjugate_t_squared));
//  X2conja(*pa, *pb);

  // Less readable version of the above that saves one complex add by using FMA instructions
  X2conjb(a, b);
  T2 minusnewb = mulminus2(cmul(a, b));                          // -newb = -2ab
  *pb = csqa(a, cfma(csq(b), conjugate_t_squared, minusnewb));   // final b = newa - newb = a^2 + (bt')^2 - newb
  (*pa).x = fma(-2.0, minusnewb.x, (*pb).x);                     // final a = newa + newb = finalb + 2 * newb
  (*pa).y = fma(2.0, minusnewb.y, -(*pb).y);                     // conjugate(final a)
}

void pairSq(u32 N, T2 *u, T2 *v, T2 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = 2 * foo(conjugate(u[i]));
      v[i] = 4 * csq(conjugate(v[i]));
    } else {
      onePairSq(&u[i], &v[i], -base_squared);
    }

    if (N == NH) {
      onePairSq(&u[i+NH/2], &v[i+NH/2], base_squared);
    }

    T2 new_base_squared = mul_t4(base_squared);
    onePairSq(&u[i+NH/4], &v[i+NH/4], -new_base_squared);

    if (N == NH) {
      onePairSq(&u[i+3*NH/4], &v[i+3*NH/4], new_base_squared);
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

  fft_HEIGHT(lds, u, smallTrig, w);
  bar();
  fft_HEIGHT(lds, v, smallTrig, w);

  // Compute trig values from scratch.  Good on GPUs with high DP throughput.
#if PREFER_DP_TO_MEM >= 2
  T2 trig = slowTrig_N(line1 + me * H, ND / NH);

  // Do a little bit of memory access and a little bit of DP math.  Good on a Radeon VII.
#elif PREFER_DP_TO_MEM == 1
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
void pairSq2_special(T2 *u, T2 base_squared) {
  u32 me = get_local_id(0);
  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (i == 0 && me == 0) {
      u[0] = 2 * foo(conjugate(u[0]));
      u[NH/2] = 4 * csq(conjugate(u[NH/2]));
    } else {
      onePairSq(&u[i], &u[NH/2+i], -base_squared);
    }
    T2 new_base_squared = mul_t4(base_squared);
    onePairSq(&u[i+NH/4], &u[NH/2+i+NH/4], -new_base_squared);
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

  new_fft_HEIGHT2_1(lds, u, smallTrig, w);

  // Compute trig values from scratch.  Good on GPUs with high DP throughput.
#if PREFER_DP_TO_MEM >= 2
  T2 trig = slowTrig_N(line + H * lowMe, ND / NH * 2);

  // Do a little bit of memory access and a little bit of DP math.  Good on a Radeon VII.
#elif PREFER_DP_TO_MEM == 1
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
    // We change the LDS halves we're using in order to enable half-bars
    revCrossLine(G_H, lds, u + NH/2, NH/2, !isSecondHalf);
  }

  bar(G_H);

  new_fft_HEIGHT2_2(lds, u, smallTrig, w);

  // Write lines u and v
  writeTailFusedLine(u, out, transPos(line, MIDDLE, WIDTH), lowMe);
}

#endif

// Copyright (C) Mihai Preda and George Woltman

#include "tailutil.cl"
#include "trig.cl"
#include "fftheight.cl"

// Klunky defines for single-wide vs. double-wide tailSquare
// Clean this up once we determine which options to make user visible
#define SINGLE_WIDE             0       // Old single-wide tailSquare
#define DOUBLE_WIDE_ONEK        0       // New single-wide tailSquare in a single kernel
#define DOUBLE_WIDE             1       // New single-wide tailSquare in two kernels


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

  X2conjb(a, b);

  T2 tmp = a;
  a = csqa(a, cmul(csq(b), conjugate_t_squared));
  b = 2 * cmul(tmp, b);

  X2conja(a, b);

  *pa = a;
  *pb = b;
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

#if SINGLE_WIDE

KERNEL(G_H) tailSquare(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[SMALL_HEIGHT];

  T2 u[NH], v[NH];

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

  fft_HEIGHT(lds, u, smallTrig, w);
  bar();
  fft_HEIGHT(lds, v, smallTrig, w);

  T2 trig = slowTrig_N(line1 + me * H, ND / NH);

  if (line1) {
    reverseLine(G_H, lds, v);
    pairSq(NH, u, v, trig, false);
    reverseLine(G_H, lds, v);
  } else {
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

#elif DOUBLE_WIDE_ONEK

// Similar to pairSq except v values are at u[NH/2]
void pairSq2(T2 *u, T2 base_squared) {
  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t4(base_squared)) {
    onePairSq(&u[i], &u[NH/2+i], -base_squared);
    onePairSq(&u[i+NH/4], &u[NH/2+i+NH/4], base_squared);
  }
}

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

KERNEL(G_H*2) tailSquare(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[SMALL_HEIGHT*2];                 // change reverse line to halve this

  T2 u[NH];

  u32 H = ND / SMALL_HEIGHT;

  u32 line_u = get_group_id(0);
  u32 me = get_local_id(0);

  u32 line_v = line_u ? H - line_u : (H / 2);
  u32 line_uv = (me < G_H) ? line_u : line_v;

  // Read lines u and v
  readTailFusedLine(in, u, line_uv, me % G_H);

#if NH == 8
  T2 w = fancyTrig_N(ND / SMALL_HEIGHT * (me % G_H));
#else
  T2 w = slowTrig_N(ND / SMALL_HEIGHT * (me % G_H), ND / NH);
#endif

  fft_HEIGHT2(lds, u, smallTrig, w);

  u32 angle = line_u + (me % G_H) * H;
  if (me >= G_H) angle += (line_u == 0) ? H / 2 : ND / NH;
  T2 trig = slowTrig_N(angle, 2 * ND / NH);

  if (line_u) {
    reverseLine2(lds, u);
    pairSq2(u, trig);
    unreverseLine2(lds, u);
  } else {
    // Line 0 and H/2 are special: they pair with themselves, line 0 is offseted by 1.
    reverse2(lds, u);
    pairSq2_special(u, trig);
    reverse2(lds, u);
  }

  if (G_H > WAVEFRONT) bar();
  fft_HEIGHT2(lds, u, smallTrig, w);

  // Write lines u and v
  writeTailFusedLine(u, out, transPos(line_uv, MIDDLE, WIDTH), me % G_H);
}

#else

// Similar to pairSq except v values are at u[NH/2]
void pairSq2(T2 *u, T2 base_squared) {
  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t4(base_squared)) {
    onePairSq(&u[i], &u[NH/2+i], -base_squared);
    onePairSq(&u[i+NH/4], &u[NH/2+i+NH/4], base_squared);
  }
}

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

KERNEL(G_H*2) tailSquareOne(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[SMALL_HEIGHT*2];                 // change reverse line to halve this

  T2 u[NH];

  u32 H = ND / SMALL_HEIGHT;

  u32 line_u = 0;
  u32 me = get_local_id(0);

  u32 line_v = H / 2;
  u32 line_uv = (me < G_H) ? line_u : line_v;

  // Read lines u and v
  readTailFusedLine(in, u, line_uv, me % G_H);

#if NH == 8
  T2 w = fancyTrig_N(ND / SMALL_HEIGHT * (me % G_H));
#else
  T2 w = slowTrig_N(ND / SMALL_HEIGHT * (me % G_H), ND / NH);
#endif

  fft_HEIGHT2(lds, u, smallTrig, w);

  T2 trig = slowTrig_N(line_uv + (me % G_H) * H, (G_H-1) * H);

  // Line 0 and H/2 are special: they pair with themselves, line 0 is offseted by 1.
  reverse2(lds, u);
  pairSq2_special(u, trig);
  reverse2(lds, u);

  bar();
  fft_HEIGHT2(lds, u, smallTrig, w);

  // Write lines u and v
  writeTailFusedLine(u, out, transPos(line_uv, MIDDLE, WIDTH), me % G_H);
}

KERNEL(G_H*2) tailSquare(P(T2) out, CP(T2) in, Trig smallTrig) {
  local T2 lds[SMALL_HEIGHT*2];                 // change reverse line to halve this

  T2 u[NH];

  u32 H = ND / SMALL_HEIGHT;

  u32 line_u = get_group_id(0) + 1;
  u32 me = get_local_id(0);

  u32 line_v = H - line_u;
  u32 line_uv = (me < G_H) ? line_u : line_v;

  // Read lines u and v
  readTailFusedLine(in, u, line_uv, me % G_H);

#if NH == 8
  T2 w = fancyTrig_N(ND / SMALL_HEIGHT * (me % G_H));
#else
  T2 w = slowTrig_N(ND / SMALL_HEIGHT * (me % G_H), ND / NH);
#endif

  fft_HEIGHT2(lds, u, smallTrig, w);

  T2 trig = slowTrig_N(line_u + (me % G_H) * H + (me / G_H) * ND / NH, 2 * ND / NH);

  reverseLine2(lds, u);
  pairSq2(u, trig);
  unreverseLine2(lds, u);

  if (G_H > WAVEFRONT) bar();
  fft_HEIGHT2(lds, u, smallTrig, w);

  // Write lines u and v
  writeTailFusedLine(u, out, transPos(line_uv, MIDDLE, WIDTH), me % G_H);
}

#endif

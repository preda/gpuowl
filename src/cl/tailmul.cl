// Copyright (C) Mihai Preda and George Woltman

#include "base.cl"
#include "tailutil.cl"
#include "trig.cl"
#include "fftheight.cl"

// Why does this alternate implementation work?  Let t' be the conjugate of t and note that t*t' = 1.
// Now consider these lines from the original implementation (comments appear alongside):
//      b = mul_by_conjugate(b, t);
//      X2(a, b);					a + bt', a - bt'
//      d = mul_by_conjugate(d, t);
//      X2(c, d);					c + dt', c - dt'
//      a = mul(a, c);					(a+bt')(c+dt') = ac + bct' + adt' + bdt'^2
//      b = mul(b, d);					(a-bt')(c-dt') = ac - bct' - adt' + bdt'^2
//      X2(a, b);					2ac + 2bdt'^2,  2bct' + 2adt'
//      b = mul(b, t);					                2bc + 2ad

void onePairMul(T2* pa, T2* pb, T2* pc, T2* pd, T2 conjugate_t_squared) {
  T2 a = *pa, b = *pb, c = *pc, d = *pd;

  X2conjb(a, b);
  X2conjb(c, d);

  T2 tmp = a;

  a = mad(a, c, mul(mul(b, d), conjugate_t_squared));
  b = mad(b, c, mul(tmp, d));

  X2conja(a, b);

  *pa = a;
  *pb = b;
  *pc = c;
  *pd = d;
}

void pairMul(u32 N, T2 *u, T2 *v, T2 *p, T2 *q, T2 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = conjugate(2 * foo2(u[i], p[i]));
      v[i] = 4 * mul(conjugate(v[i]), conjugate(q[i]));
    } else {
      onePairMul(&u[i], &v[i], &p[i], &q[i], -base_squared);
    }

    if (N == NH) {
      onePairMul(&u[i+NH/2], &v[i+NH/2], &p[i+NH/2], &q[i+NH/2], base_squared);
    }

    // T2 new_base_squared = mul(base_squared, U2(0, -1));
    T2 new_base_squared = U2(IM(base_squared), -RE(base_squared));
    onePairMul(&u[i+NH/4], &v[i+NH/4], &p[i+NH/4], &q[i+NH/4], -new_base_squared);

    if (N == NH) {
      onePairMul(&u[i+3*NH/4], &v[i+3*NH/4], &p[i+3*NH/4], &q[i+3*NH/4], new_base_squared);
    }
  }
}

KERNEL(G_H) tailMul(P(T2) out, CP(T2) in, CP(T2) a, Trig smallTrig, BigTab tailTrig) {
  local T2 lds[SMALL_HEIGHT / 2];

  T2 u[NH], v[NH];
  T2 p[NH], q[NH];

  u32 W = SMALL_HEIGHT;
  u32 H = ND / W;

  u32 line1 = get_group_id(0);
  u32 line2 = line1 ? H - line1 : (H / 2);
  u32 memline1 = transPos(line1, MIDDLE, WIDTH);
  u32 memline2 = transPos(line2, MIDDLE, WIDTH);
    
#if MUL_LOW
  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  read(G_H, NH, p, a, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, a, memline2 * SMALL_HEIGHT);
  fft_HEIGHT(lds, u, smallTrig);
  bar();
  fft_HEIGHT(lds, v, smallTrig);
#else
  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);
  readTailFusedLine(a, p, line1);
  readTailFusedLine(a, q, line2);
  fft_HEIGHT(lds, u, smallTrig);
  bar();
  fft_HEIGHT(lds, v, smallTrig);
  bar();
  fft_HEIGHT(lds, p, smallTrig);
  bar();
  fft_HEIGHT(lds, q, smallTrig);
#endif

  u32 me = get_local_id(0);

  if (line1 == 0) {

#if TAIL_TAB
    T2 trig1 = tailTrig[me];
#else
    T2 trig1 = slowTrig_N(me * H, ND / NH, NULL);     // slowTrig_2SH(2 * me, SMALL_HEIGHT / 2, TRIG_2SH)
#endif

    T2 trig2 = fancyMulTrig(trig1, tailTrig[G_H]);

    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, trig1, true);
    reverse(G_H, lds, u + NH/2, true);

    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, trig2, false);
    reverse(G_H, lds, v + NH/2, false);
  } else {    
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);

#if TAIL_TAB
    T2 trig = fancyMulTrig(tailTrig[me], tailTrig[G_H + line1]);
#else
    T2 trig = slowTrig_N(line1 + me * H, ND / NH, NULL);
#endif

    pairMul(NH, u, v, p, q, trig, false);
    reverseLine(G_H, lds, v);
  }

  bar();
  fft_HEIGHT(lds, v, smallTrig);
  write(G_H, NH, v, out, memline2 * SMALL_HEIGHT);

  bar();
  fft_HEIGHT(lds, u, smallTrig);
  write(G_H, NH, u, out, memline1 * SMALL_HEIGHT);
}

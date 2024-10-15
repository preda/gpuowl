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

  a = cfma(a, c, cmul(cmul(b, d), conjugate_t_squared));
  b = cfma(b, c, cmul(tmp, d));

  X2conja(a, b);

  *pa = a;
  *pb = b;
}

void pairMul(u32 N, T2 *u, T2 *v, T2 *p, T2 *q, T2 base_squared, bool special) {
  u32 me = get_local_id(0);

  for (i32 i = 0; i < NH / 4; ++i, base_squared = mul_t8(base_squared)) {
    if (special && i == 0 && me == 0) {
      u[i] = conjugate(2 * foo2(u[i], p[i]));
      v[i] = 4 * cmul(conjugate(v[i]), conjugate(q[i]));
    } else {
      onePairMul(&u[i], &v[i], &p[i], &q[i], -base_squared);
    }

    if (N == NH) {
      onePairMul(&u[i+NH/2], &v[i+NH/2], &p[i+NH/2], &q[i+NH/2], base_squared);
    }

    T2 new_base_squared = mul_t4(base_squared);
    onePairMul(&u[i+NH/4], &v[i+NH/4], &p[i+NH/4], &q[i+NH/4], -new_base_squared);

    if (N == NH) {
      onePairMul(&u[i+3*NH/4], &v[i+3*NH/4], &p[i+3*NH/4], &q[i+3*NH/4], new_base_squared);
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
    
  readTailFusedLine(in, u, line1);
  readTailFusedLine(in, v, line2);

#if MUL_LOW
  read(G_H, NH, p, a, memline1 * SMALL_HEIGHT);
  read(G_H, NH, q, a, memline2 * SMALL_HEIGHT);
  fft_HEIGHT(lds, u, smallTrig);
  bar();
  fft_HEIGHT(lds, v, smallTrig);
#else
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
  T2 trig = slowTrig_N(line1 + me * H, ND / NH);

  if (line1) {
    reverseLine(G_H, lds, v);
    reverseLine(G_H, lds, q);
    pairMul(NH, u, v, p, q, trig, false);
    reverseLine(G_H, lds, v);
  } else {
    reverse(G_H, lds, u + NH/2, true);
    reverse(G_H, lds, p + NH/2, true);
    pairMul(NH/2, u,  u + NH/2, p, p + NH/2, trig, true);
    reverse(G_H, lds, u + NH/2, true);

    T2 trig2 = cmulFancy(trig, TAILT);
    reverse(G_H, lds, v + NH/2, false);
    reverse(G_H, lds, q + NH/2, false);
    pairMul(NH/2, v,  v + NH/2, q, q + NH/2, trig2, false);
    reverse(G_H, lds, v + NH/2, false);
  }

  bar();
  fft_HEIGHT(lds, v, smallTrig);
  bar();
  fft_HEIGHT(lds, u, smallTrig);
  write(G_H, NH, v, out, memline2 * SMALL_HEIGHT);
  write(G_H, NH, u, out, memline1 * SMALL_HEIGHT);
}

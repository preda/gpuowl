// Copyright (C) Mihai Preda

#include "trig.cl"

void fft2(T2* u) { X2(u[0], u[1]); }

#if MIDDLE == 3
#include "fft3.cl"
#elif MIDDLE == 4
#include "fft4.cl"
#elif MIDDLE == 5
#include "fft5.cl"
#elif MIDDLE == 6
#include "fft6.cl"
#elif MIDDLE == 7
#include "fft7.cl"
#elif MIDDLE == 8
#include "fft8.cl"
#elif MIDDLE == 9
#include "fft9.cl"
#elif MIDDLE == 10
#include "fft10.cl"
#elif MIDDLE == 11
#include "fft11.cl"
#elif MIDDLE == 12
#include "fft12.cl"
#elif MIDDLE == 13
#include "fft13.cl"
#elif MIDDLE == 14
#include "fft14.cl"
#elif MIDDLE == 15
#include "fft15.cl"
#endif

void fft_MIDDLE(T2 *u) {
#if MIDDLE == 1
  // Do nothing
#elif MIDDLE == 2
  fft2(u);
#elif MIDDLE == 3
  fft3(u);
#elif MIDDLE == 4
  fft4(u);
#elif MIDDLE == 5
  fft5(u);
#elif MIDDLE == 6
  fft6(u);
#elif MIDDLE == 7
  fft7(u);
#elif MIDDLE == 8
  fft8(u);
#elif MIDDLE == 9
  fft9(u);
#elif MIDDLE == 10
  fft10(u);
#elif MIDDLE == 11
  fft11(u);
#elif MIDDLE == 12
  fft12(u);
#elif MIDDLE == 13
  fft13(u);
#elif MIDDLE == 14
  fft14(u);
#elif MIDDLE == 15
  fft15(u);
#else
#error UNRECOGNIZED MIDDLE
#endif
}

// Apply the twiddles needed after fft_MIDDLE and before fft_HEIGHT in forward FFT.
// Also used after fft_HEIGHT and before fft_MIDDLE in inverse FFT.

#define WADD(i, w) u[i] = mul(u[i], w)
#define WSUB(i, w) u[i] = mul_by_conjugate(u[i], w)

#define WADDF(i, w) u[i] = fancyMulTrig(u[i], w)
#define WSUBF(i, w) u[i] = fancyMulTrig(u[i], conjugate(w))
// mul(u[i], U2(w.x + 1, w.y))
//

void middleMul(T2 *u, u32 s, Trig trig, BigTab TRIG_BH) {
  assert(s < SMALL_HEIGHT);
  if (MIDDLE == 1) { return; }

  T2 w = trig[s];
  // slowTrig_BH(s, SMALL_HEIGHT, TRIG_BH);
  WADDF(1, w);

#if MM_CHAIN == 1 && MIDDLE >= 5

  u32 n = (MIDDLE - 1) / 3;
  u32 m = MIDDLE - 2 * n - 1;
  u32 midpoint = m + n;

  T2 base1 = slowTrig_BH(s * midpoint, SMALL_HEIGHT * midpoint, TRIG_BH);
      // trig[s * (MIDDLE - 1) + (midpoint - 1)];
      //
  T2 base2 = base1;
  WADD(midpoint, base1);
  for (i32 i = 1; i <= n; ++i) {
    base1 = fancyMulTrig(base1, conjugate(w));
    WADD(midpoint - i, base1);

    base2 = fancyMulTrig(base2, w);
    WADD(midpoint + i, base2);
  }

#elif MM_CHAIN == 0 || MIDDLE < 5
  u32 m = MIDDLE;
#else
#error MM_CHAIN must be 0 or 1
#endif

  if (m <= 2) { return; }
  T2 base = fancySqUpdate(w);
  WADDF(2, base);
  base.x += 1;

  for (i32 i = 3; i < m; ++i) {
    base = fancyMulTrig(base, w);
    WADD(i, base);
  }
}

void middleMul2(T2 *u, u32 x, u32 y, double factor, Trig trig, BigTab TRIG_BHW) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);

#if MM2_CHAIN > 1
#error MM2_CHAIN must be 0 or 1
#endif

  if (MIDDLE <= 2) { // MIDDLE in [1, 2]
    T2 w = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE, TRIG_BHW);
    T2 base = slowTrig_N(x * y, ND / MIDDLE, TRIG_BHW) * factor;
    for (int i = 0; i < MIDDLE; ++i) {
      WADD(i, base);
      base = mul(base, w);
    }

  } else if (MIDDLE <= 4) { // MIDDLE in [3, 4]
    T2 w = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE, TRIG_BHW);
    T2 base = slowTrig_N(x * y + x * SMALL_HEIGHT, ND / MIDDLE * 2, TRIG_BHW) * factor;
    WADD(0, base);
    WADD(1, base);
    WADD(2, base);
    if (MIDDLE == 4) { WADD(3, base); }
    WSUB(0, w);
    WADD(2, w);
    if (MIDDLE == 4) { WADD(3, w); WADD(3, w); }

  } else { // MIDDLE >= 5
    // T2 w = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE, TRIG_BHW);
    T2 w = trig[SMALL_HEIGHT + x];

#if MM2_CHAIN == 0
    u32 cnt = 1;
    for (u32 start = 0, sz = (MIDDLE - start + cnt - 1) / cnt; cnt > 0; --cnt, start += sz) {
      if (start + sz > MIDDLE) { --sz; }
      u32 n = (sz - 1) / 2;
      u32 mid = start + n;

      T2 base1 = slowTrig_N(x * y + x * SMALL_HEIGHT * mid, ND / MIDDLE * (mid + 1), TRIG_BHW) * factor;
      WADD(mid, base1);

      T2 base2 = base1;
      for (u32 i = 1; i <= n; ++i) {
        base1 = fancyMulTrig(base1, conjugate(w));
        WADD(mid - i, base1);

        base2 = fancyMulTrig(base2, w);
        WADD(mid + i, base2);
      }
      if (!(sz & 1)) {
        base2 = fancyMulTrig(base2, w);
        WADD(mid + n + 1, base2);
      }
    }
#else
    T2 base;
    for (u32 i = 1; i < MIDDLE; i += 3) {
      base = slowTrig_N(x * y + x * SMALL_HEIGHT * i, ND / MIDDLE * (i + 1), TRIG_BHW) * factor;
      WADD(i-1, base);
      WADD(i,   base);
      if (i + 1 < MIDDLE) { WADD(i+1, base); }
    }
    if (MIDDLE % 3 == 1) { WADD(MIDDLE-1, base); }
    for (u32 i = 0; i + 1 < MIDDLE; i += 3) { WSUBF(i, w); }
    for (u32 i = 2; i < MIDDLE; i += 3) { WADDF(i, w); }
    if (MIDDLE % 3 == 1) { WADDF(MIDDLE-1, w); WADDF(MIDDLE-1, w); }
#endif
  }
}

#undef WADD
#undef WADDF
#undef WSUB
#undef WSUBF

// Do a partial transpose during fftMiddleIn/Out
// The AMD OpenCL optimization guide indicates that reading/writing T values will be more efficient
// than reading/writing T2 values.  This routine lets us try both versions.

void middleShuffle(local T *lds, T2 *u, u32 workgroupSize, u32 blockSize) {
  u32 me = get_local_id(0);
  if (MIDDLE <= 8) {
    local T *p = lds + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    for (int i = 0; i < MIDDLE; ++i) { p[i * workgroupSize] = u[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].x = lds[me + workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p[i * workgroupSize] = u[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { u[i].y = lds[me + workgroupSize * i]; }
  } else {
    local int *p1 = ((local int*) lds) + (me % blockSize) * (workgroupSize / blockSize) + me / blockSize;
    local int *p2 = (local int*) lds;
    int4 *pu = (int4 *)u;

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].x; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].x = p2[me + workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].y; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].y = p2[me + workgroupSize * i]; }
    bar();

    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].z; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].z = p2[me + workgroupSize * i]; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { p1[i * workgroupSize] = pu[i].w; }
    bar();
    for (int i = 0; i < MIDDLE; ++i) { pu[i].w = p2[me + workgroupSize * i]; }
  }
}

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
#define WSUB(i, w) u[i] = mul_by_conjugate(u[i], w);

void middleMul(T2 *u, u32 s, Trig trig, BigTab TRIG_BH) {
  assert(s < SMALL_HEIGHT);
  if (MIDDLE == 1) { return; }

#if MM_CHAIN == 4
    for (int i = 1; i < MIDDLE; ++i) {
      WADD(i, trig[s * (MIDDLE - 1) + (i - 1)]);
      // WADD(i, trig[s + (i - 1) * SMALL_HEIGHT]);
      // WADD(i, slowTrig_BH(s * i, SMALL_HEIGHT * i, TRIG_BH));
    }
#else

  T2 w = trig[s * (MIDDLE - 1)];
      // slowTrig_BH(s, SMALL_HEIGHT, TRIG_BH);

#if MM_CHAIN == 3
// This is our slowest version - used when we are extremely worried about round off error.
// Maximum multiply chain length is 1.
  WADD(1, w);
  if ((MIDDLE - 2) % 3) {
    T2 base = slowTrig_BH(s * 2, SMALL_HEIGHT * 2, TRIG_BH);
    WADD(2, base);
    if ((MIDDLE - 2) % 3 == 2) {
      WADD(3, base);
      WADD(3, w);
    }
  }
  for (i32 i = (MIDDLE - 2) % 3 + 3; i < MIDDLE; i += 3) {
    T2 base = slowTrig_BH(s * i, SMALL_HEIGHT * i, TRIG_BH);
    WADD(i - 1, base);
    WADD(i, base);
    WADD(i + 1, base);
    WSUB(i - 1, w);
    WADD(i + 1, w);
  }

#elif MM_CHAIN == 1 && MIDDLE >= 5
  WADD(1, w);

#if 1
  u32 n = (MIDDLE - 1) / 3;
  u32 m = MIDDLE - 2 * n - 3;
  u32 midpoint = 2 + m + n;
#else
  u32 n = (MIDDLE - 4) / 3;
  u32 m = MIDDLE - 2 * n - 3;
  u32 midpoint = 2 + m + n;
#endif

  T2 base = trig[s * (MIDDLE - 1) + (midpoint - 1)];
      // slowTrig_BH(s * midpoint, SMALL_HEIGHT * midpoint, TRIG_BH);
  T2 base2 = base;
  WADD(midpoint, base);
  for (i32 i = 1; i <= n; ++i) {
    base = mul_by_conjugate(base, w);
    WADD(midpoint - i, base);
    base2 = mul(base2, w);
    WADD(midpoint + i, base2);
  }

  base = w;
  for (i32 i = 2; i < 2 + m; ++i) {
    base = mul(base, w);
    WADD(i, base);
  }

#elif MM_CHAIN == 2
// This is our second and third fastest versions - used when we are somewhat worried about round off error.
// Maximum multiply chain length is MIDDLE/2 or MIDDLE/4.
  WADD(1, w);
  WADD(2, sq(w));
  i32 group_start, group_size;
  for (group_start = 3; group_start < MIDDLE; group_start += group_size) {

#if MIDDLE > 4
    group_size = (group_start == 3 ? (MIDDLE - 3) / 2 : MIDDLE - group_start);
#else
    group_size = MIDDLE - 3;
#endif

    i32 midpoint = group_start + group_size / 2;
    T2 base = slowTrig_BH(s * midpoint, SMALL_HEIGHT * midpoint, TRIG_BH);
    T2 base2 = base;
    WADD(midpoint, base);
    for (i32 i = 1; i <= group_size / 2; ++i) {
      base = mul_by_conjugate(base, w);
      WADD(midpoint - i, base);
      if (i == group_size / 2 && (group_size & 1) == 0) break;
      base2 = mul(base2, w);
      WADD(midpoint + i, base2);
    }
  }

#else // MM_CHAIN=0
  WADD(1, w);
  T2 base = sq(w);
  for (i32 i = 2; i < MIDDLE; ++i) {
    WADD(i, base);
    base = mul(base, w);
  }
#endif
#endif
}

void middleMul2(T2 *u, u32 x, u32 y, double factor, BigTab TRIG_BHW) {
  assert(x < WIDTH);
  assert(y < SMALL_HEIGHT);
  T2 w = slowTrig_N(x * SMALL_HEIGHT, ND / MIDDLE, TRIG_BHW);

#if MM2_CHAIN == 3
// This is our slowest version - used when we are extremely worried about round off error.
// Maximum multiply chain length is 1.
  if (MIDDLE % 3) {
    T2 base = slowTrig_N(x * y, ND / MIDDLE, TRIG_BHW) * factor;
    WADD(0, base);
    if (MIDDLE % 3 == 2) {
      WADD(1, base);
      WADD(1, w);
    }
  }
  for (i32 i = MIDDLE % 3 + 1; i < MIDDLE; i += 3) {
    T2 base = slowTrig_N(x * SMALL_HEIGHT * i + x * y, ND / MIDDLE * (i + 1), TRIG_BHW) * factor;
    WADD(i - 1, base);
    WADD(i, base);
    WADD(i + 1, base);
    WSUB(i - 1, w);
    WADD(i + 1, w);
  }

#elif MM2_CHAIN == 1 || MM2_CHAIN == 2

// This is our second and third fastest versions - used when we are somewhat worried about round off error.
// Maximum multiply chain length is MIDDLE/2 or MIDDLE/4.
  i32 group_size = 0;
  for (i32 group_start = 0; group_start < MIDDLE; group_start += group_size) {
#if MM2_CHAIN == 2
    group_size = (group_start == 0 ? MIDDLE / 2 : MIDDLE - group_start);
#else
    group_size = MIDDLE;
#endif
    i32 midpoint = group_start + group_size / 2;
    T2 base = slowTrig_N(x * SMALL_HEIGHT * midpoint + x * y, ND / MIDDLE * (midpoint + 1), TRIG_BHW) * factor;
    T2 base2 = base;
    WADD(midpoint, base);
    for (i32 i = 1; i <= group_size / 2; ++i) {
      base = mul_by_conjugate(base, w);
      WADD(midpoint - i, base);
      if (i == group_size / 2 && (group_size & 1) == 0) break;
      base2 = mul(base2, w);
      WADD(midpoint + i, base2);
    }
  }

#else

// This is our fastest version - used when we are not worried about round off error.
// Maximum multiply chain length equals MIDDLE.
  T2 base = slowTrig_N(x * y, ND/MIDDLE, TRIG_BHW) * factor;
  for (i32 i = 0; i < MIDDLE; ++i) {
    WADD(i, base);
    base = mul(base, w);
  }

#endif

}

#undef WADD
#undef WSUB

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

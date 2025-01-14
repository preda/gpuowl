// Copyright (C) Mihai Preda

#include "base.cl"
#include "fftbase.cl"
#include "middle.cl"

#if SMALL_HEIGHT != 256 && SMALL_HEIGHT != 512 && SMALL_HEIGHT != 1024 && SMALL_HEIGHT != 4096
#error SMALL_HEIGHT must be one of: 256, 512, 1024, 4096
#endif

u32 transPos(u32 k, u32 middle, u32 width) { return k / width + k % width * middle; }

void fft_NH(T2 *u) {
#if NH == 4
  fft4(u);
#elif NH == 8
  fft8(u);
#else
#error NH
#endif
}

#if BCAST && (HEIGHT <= 1024)

void fft_HEIGHT(local T2 *lds, T2 *u, Trig trig, T2 w) {
  for (u32 s = 1; s < SMALL_HEIGHT / NH; s *= NH) {
    if (s > 1) { bar(); }
    fft_NH(u);
    w = bcast(w, s);

#if NH == 8
    chainMul8(u, w);
#else
    chainMul4(u, w);
#endif

    shufl(SMALL_HEIGHT / NH, lds,  u, NH, s);
  }
  fft_NH(u);
}

void fft_HEIGHT2(local T2 *lds, T2 *u, Trig trig, T2 w) {
  u32 WG = SMALL_HEIGHT / NH;
  for (u32 s = 1; s < SMALL_HEIGHT / NH; s *= NH) {
    if (s > 1) { bar(WG); }
    fft_NH(u);
    w = bcast(w, s);

#if NH == 8
    chainMul8(u, w);
#else
    chainMul4(u, w);
#endif

    shufl2(SMALL_HEIGHT / NH, lds,  u, NH, s);
  }
  fft_NH(u);
}

#else

void fft_HEIGHT(local T2 *lds, T2 *u, Trig trig, T2 w) {
  u32 me = get_local_id(0);

#if !UNROLL_H
  __attribute__((opencl_unroll_hint(1)))
#endif

  for (u32 s = 1; s < SMALL_HEIGHT / NH; s *= NH) {
    if (s > 1) { bar(); }
    fft_NH(u);
    tabMul(SMALL_HEIGHT / NH, trig, u, NH, s, me);
    shufl(SMALL_HEIGHT / NH, lds,  u, NH, s);
  }
  fft_NH(u);
}

void fft_HEIGHT2(local T2 *lds, T2 *u, Trig trig, T2 w) {
  u32 me = get_local_id(0);
  u32 WG = SMALL_HEIGHT / NH;

#if !UNROLL_H
  __attribute__((opencl_unroll_hint(1)))
#endif

  for (u32 s = 1; s < WG; s *= NH) {
    if (s > 1) { bar(WG); }
    fft_NH(u);
    tabMul(WG, trig, u, NH, s, me % WG);
    shufl2(WG, lds,  u, NH, s);
  }
  fft_NH(u);
}

#endif



void new_fft_HEIGHT2(local T2 *lds, T2 *u, Trig trig, T2 w, int callnum) {
  u32 WG = SMALL_HEIGHT / NH;
  u32 me = get_local_id(0);
  // This line mimics shufl2 -- partition lds into halves
  local T2* partitioned_lds = lds + (me / WG) * SMALL_HEIGHT / 2;
  me = me % WG;

// Custom code for various SMALL_HEIGHT values

#if SMALL_HEIGHT == 512 && NH == 8 && !BCAST && CLEAN == 1

// Custom code for SMALL_HEIGHT=512, NH=8

  T preloads[8];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul_trig(WG, trig, preloads, 1, me);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul(WG, partitioned_lds, trig, preloads, u, 1, me);
  shufl2(WG, lds, u, NH, 1);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul_fft8(WG, partitioned_lds, trig, preloads, u, 1, me, 1);
  partial_tabMul(WG, partitioned_lds, trig, preloads, u, 8, me);
  bar(WG);
  shufl2(WG, lds, u, NH, 8);

  // Finish second tabMul and perform final fft8.
  finish_tabMul_fft8(WG, partitioned_lds, trig, preloads, u, 8, me, 1);

#else

  // Old version
  fft_HEIGHT2(lds, u, trig, w);

#endif
}

void new_fft_HEIGHT2_1(local T2 *lds, T2 *u, Trig trig, T2 w)  { new_fft_HEIGHT2(lds, u, trig, w, 1); }
void new_fft_HEIGHT2_2(local T2 *lds, T2 *u, Trig trig, T2 w)  { new_fft_HEIGHT2(lds, u, trig, w, 2); }


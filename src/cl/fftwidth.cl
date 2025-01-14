// Copyright (C) Mihai Preda

#include "fftbase.cl"

#if WIDTH != 256 && WIDTH != 512 && WIDTH != 1024 && WIDTH != 4096 && WIDTH != 625
#error WIDTH must be one of: 256, 512, 1024, 4096, 625
#endif

void fft_NW(T2 *u) {
#if NW == 4
  fft4(u);
#elif NW == 5
  fft5(u);
#elif NW == 8
  fft8(u);
#else
#error NW
#endif
}

#if BCAST && (WIDTH <= 1024)

void fft_WIDTH(local T2 *lds, T2 *u, Trig trig) {
  u32 me = get_local_id(0);
#if NW == 8
  T2 w = fancyTrig_N(ND / WIDTH * me);
#else
  T2 w = slowTrig_N(ND / WIDTH * me, ND / NW);
#endif

  for (u32 s = 1; s < WIDTH / NW; s *= NW) {
    if (s > 1) { bar(); }
    fft_NW(u);
    w = bcast(w, s);

#if NW == 8
    chainMul8(u, w);
#else
    chainMul4(u, w);
#endif

    shufl( WIDTH / NW, lds,  u, NW, s);
  }
  fft_NW(u);
}

#else

void fft_WIDTH(local T2 *lds, T2 *u, Trig trig) {
  u32 me = get_local_id(0);

#if !UNROLL_W
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WIDTH / NW; s *= NW) {
    if (s > 1) { bar(); }
    fft_NW(u);
    tabMul(WIDTH / NW, trig, u, NW, s, me);
    shufl( WIDTH / NW, lds,  u, NW, s);
  }
  fft_NW(u);
}

#endif





// New fft_WIDTH that uses more FMA instructions than the old fft_WIDTH.
// The tabMul after fft8 only does a partial complex multiply, saving a mul-by-cosine for the next fft8 using FMA instructions.
// To maximize FMA opportunities we precompute tig values as cosine and sine/cosine rather than cosine and sine.
// The downside is sine/cosine cannot be computed with chained multiplies.

void new_fft_WIDTH(local T2 *lds, T2 *u, Trig trig, int callnum) {
  u32 WG = WIDTH / NW;
  u32 me = get_local_id(0);

// Custom code for various WIDTH values

#if WIDTH == 512 && NW == 8 && !BCAST && CLEAN == 1 && UNROLL_W >= 3

// Custom code for WIDTH=512, NW=8

  T preloads[8];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul_trig(WG, trig, preloads, 1, me);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul(WG, lds, trig, preloads, u, 1, me);
  shufl(WG, lds, u, NW, 1);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul_fft8(WG, lds, trig, preloads, u, 1, me, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul(WG, lds, trig, preloads, u, 8, me);
  bar();
  shufl(WG, lds, u, NW, 8);

  // Finish second tabMul and perform final fft8.
  finish_tabMul_fft8(WG, lds, trig, preloads, u, 8, me, 0);  // We'd rather set save_one_more_mul to 1

#else

  // Old version
  fft_WIDTH(lds, u, trig);

#endif
}

void new_fft_WIDTH1(local T2 *lds, T2 *u, Trig trig) { new_fft_WIDTH(lds, u, trig, 1); }
void new_fft_WIDTH2(local T2 *lds, T2 *u, Trig trig) { new_fft_WIDTH(lds, u, trig, 2); }


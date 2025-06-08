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

#if FFT_VARIANT_W == 0

#if WIDTH > 1024
#error FFT_VARIANT_W == 0 only supports WIDTH <= 1024
#endif
#if !AMDGPU
#error FFT_VARIANT_W == 0 only supported by AMD GPUs
#endif

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

    chainMul(NW, u, w, 0);

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

#if WIDTH == 256 && NW == 4 && FFT_VARIANT_W == 2

// Custom code for WIDTH=256, NW=4

  T preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(WG, trig, preloads, 1, me);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(WG, lds, trig, preloads, u, 1, me);
  shufl(WG, lds, u, NW, 1);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(WG, lds, trig, preloads, u, 1, me, 1);
  partial_tabMul4(WG, lds, trig, preloads, u, 4, me);
  bar(WG);
  shufl(WG, lds, u, NW, 4);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(WG, lds, trig, preloads, u, 4, me, 1);
  partial_tabMul4(WG, lds, trig, preloads, u, 16, me);
  bar(WG);
  shufl(WG, lds, u, NW, 16);

  // Finish third tabMul and perform final fft4.
  finish_tabMul4_fft4(WG, lds, trig, preloads, u, 16, me, 1);

#elif WIDTH == 512 && NW == 8 && FFT_VARIANT_W == 2

// Custom code for WIDTH=512, NW=8

  T preloads[10];             // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8;               // Skip past old FFT_width trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(WG, trig, preloads, 1, me);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(WG, lds, trig, preloads, u, 1, me);
  shufl(WG, lds, u, NW, 1);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(WG, lds, trig, preloads, u, 1, me, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(WG, lds, trig, preloads, u, 8, me);
  bar();
  shufl(WG, lds, u, NW, 8);

  // Finish second tabMul and perform final fft8.
  finish_tabMul8_fft8(WG, lds, trig, preloads, u, 8, me, 0);  // We'd rather set save_one_more_mul to 1

#elif WIDTH == 1024 && NW == 4 && FFT_VARIANT_W == 2

// Custom code for WIDTH=1024, NW=4

  T preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(WG, trig, preloads, 1, me);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(WG, lds, trig, preloads, u, 1, me);
  shufl(WG, lds, u, NW, 1);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(WG, lds, trig, preloads, u, 1, me, 1);
  partial_tabMul4(WG, lds, trig, preloads, u, 4, me);
  bar(WG);
  shufl(WG, lds, u, NW, 4);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(WG, lds, trig, preloads, u, 4, me, 1);
  partial_tabMul4(WG, lds, trig, preloads, u, 16, me);
  bar(WG);
  shufl(WG, lds, u, NW, 16);

  // Finish the third tabMul and perform fourth fft4.  Do fourth partial tabMul and shufl.
  finish_tabMul4_fft4(WG, lds, trig, preloads, u, 16, me, 1);
  partial_tabMul4(WG, lds, trig, preloads, u, 64, me);
  bar(WG);
  shufl(WG, lds, u, NW, 64);

  // Finish fourth tabMul and perform final fft4.
  finish_tabMul4_fft4(WG, lds, trig, preloads, u, 64, me, 1);

#elif WIDTH == 4096 && NW == 8 && FFT_VARIANT_W == 2

// Custom code for WIDTH=4K, NW=8

  T preloads[10];             // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8;               // Skip past old FFT_width trig values to the !save_one_more_mul trig values

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(WG, trig, preloads, 1, me);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(WG, lds, trig, preloads, u, 1, me);
  shufl(WG, lds, u, NW, 1);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(WG, lds, trig, preloads, u, 1, me, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(WG, lds, trig, preloads, u, 8, me);
  bar();
  shufl(WG, lds, u, NW, 8);

  // Finish the second tabMul and perform third fft8.  Do third partial tabMul and shufl.
  finish_tabMul8_fft8(WG, lds, trig, preloads, u, 8, me, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(WG, lds, trig, preloads, u, 64, me);
  bar();
  shufl(WG, lds, u, NW, 64);

  // Finish third tabMul and perform final fft8.
  finish_tabMul8_fft8(WG, lds, trig, preloads, u, 64, me, 0);  // We'd rather set save_one_more_mul to 1

#else

  // Old version
  fft_WIDTH(lds, u, trig);

#endif
}

void new_fft_WIDTH1(local T2 *lds, T2 *u, Trig trig) { new_fft_WIDTH(lds, u, trig, 1); }
void new_fft_WIDTH2(local T2 *lds, T2 *u, Trig trig) { new_fft_WIDTH(lds, u, trig, 2); }


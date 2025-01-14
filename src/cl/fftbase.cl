// Copyright (C) Mihai Preda

#include "fft4.cl"
#include "fft8.cl"
#include "trig.cl"
// #include "math.cl"

#if BCAST

int bcast4(int x)  { return __builtin_amdgcn_mov_dpp(x, 0, 0xf, 0xf, false); }
int bcast8(int x)  { return __builtin_amdgcn_ds_swizzle(x, 0x0018); }
int bcast16(int x) { return __builtin_amdgcn_ds_swizzle(x, 0x0010); }
int bcast64(int x) { return __builtin_amdgcn_readfirstlane(x); }

int bcastAux(int x, u32 span) {
  return span == 4 ? bcast4(x) : span == 8 ? bcast8(x) : span == 16 ? bcast16(x) : span == 64 ? bcast64(x) : x;
}

T2 bcast(T2 src, u32 span) {
  int4 s = as_int4(src);
  for (int i = 0; i < 4; ++i) { s[i] = bcastAux(s[i], span); }
  return as_double2(s);
}

void chainMul4(T2 *u, T2 w) {
  u[1] = cmul(u[1], w);

  T2 base = csqTrig(w);
  u[2] = cmul(u[2], base);

  double a = 2 * base.y;
  base = U2(fma(a, -w.y, w.x), fma(a, w.x, -w.y));
  u[3] = cmul(u[3], base);
}

void chainMul8(T2 *u, T2 w) {
  u[1] = cmulFancy(u[1], w);

#if 1
  T2 base = 2 * U2(- w.y * w.y, fma(w.x, w.y, w.y));
  u[2] = cmulFancy(u[2], base);
#else
  T2 base = U2(fma(-2 * w.y, w.y, 1), 2 * fma(w.x, w.y, w.y));
  u[2] = cmul(u[2], base);
#endif

  double a = 2 * base.y;
  // base = U2(fma(a, -w.y, w.x + 1), fma(a, w.x, a - w.y));
  base = U2(fma(a, -w.y, w.x + 1), fma(a, w.x, a - w.y));

  for (int i = 3; i < 8; ++i) {
    u[i] = cmul(u[i], base);
    base = cmulFancy(base, w);
  }
}

#endif

void shuflBigLDS(u32 WG, local T2 *lds, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);

  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i]; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i] = lds[i * WG + me]; }
}

void shufl(u32 WG, local T2 *lds2, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  local T* lds = (local T*) lds2;

  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i].x; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].x = lds[i * WG + me]; }
  bar();
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i].y; }
  bar();
  for (u32 i = 0; i < n; ++i) { u[i].y = lds[i * WG + me]; }
}

// Shufl two simultaneous FFT_HEIGHTs.  Needed for tailSquared where u and v are computed simultaneously in different threads.
// NOTE:  It is very important for this routine to use lds memory in coordination with reverseLine2 and unreverseLine2.
// Failure to do so would result in the need for more bar() calls.  Specifically, the u values are stored in the upper half
// of lds memory (first SMALL_HEIGHT T2 values).  The v values are stored in the lower half of lds memory (next SMALL_HEIGHT T2 values).
void shufl2(u32 WG, local T2 *lds2, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);

  // Partition lds memory into upper and lower halves
  assert(WG == G_H);

  // Accessing lds memory as doubles is faster than T2 accesses
  local T* lds = ((local T*) lds2) + (me / WG) * SMALL_HEIGHT;

  me = me % WG;
  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);
  
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i].x; }
  bar(WG);
  for (u32 i = 0; i < n; ++i) { u[i].x = lds[i * WG + me]; }
  bar(WG);
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = u[i].y; }
  bar(WG);
  for (u32 i = 0; i < n; ++i) { u[i].y = lds[i * WG + me]; }  
}

void tabMul(u32 WG, Trig trig, T2 *u, u32 n, u32 f, u32 me) {
#if 0
  u32 p = me / f * f;
#else
  u32 p = me & ~(f - 1);
#endif

#if 0
  T2 w = slowTrig_N(ND / n / WG * p, ND / n);
  T2 base = w;
  for (int i = 1; i < n; ++i) {
    u[i] = cmul(u[i], w);
    w = cmul(w, base);
  }
#endif

  T2 w = trig[p];

  if (n >= 8) {
    u[1] = cmulFancy(u[1], w);
  } else {
    u[1] = cmul(u[1], w);
  }

// Theoretically, maximum accuracy.  Uses memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.
#if CLEAN == 1                // Radeon VII loves this case, in fact it is faster than the CLEAN == 0 case.  nVidia Titan V hates this case.

  for (u32 i = 2; i < n; ++i) {
    T2 base = trig[(i-1)*WG + p];
    u[i] = cmul(u[i], base);
  }

// Original CLEAN==1, saves one cmul at the cost of a memory access.  I see little use for this case.
#elif CLEAN == 1
  T2 base = trig[WG + p];

  if (n >= 8) {
    for (u32 i = 2; i < n; ++i) {
      u[i] = cmul(u[i], base);
      base = cmulFancy(base, w);
    }
  } else {
    for (u32 i = 2; i < n; ++i) {
      u[i] = cmul(u[i], base);
      base = cmul(base, w);
    }
  }

// This code uses chained complex multiplies which could be faster on GPUs with great DP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice this is more accurate (at least when n==8) than reading precomputed
// values from memory.  Perhaps chained Fancy muls are the reason.
#elif CLEAN == 0
  if (n >= 8) {
    T2 base = csqTrigFancy(w);
    u[2] = cmulFancy(u[2], base);
    base = ccubeTrigFancy(base, w);
    for (u32 i = 3; i < n; ++i) {
      u[i] = cmul(u[i], base);
      base = cmulFancy(base, w);
    }
  } else {
    T2 base = csqTrig(w);
    u[2] = cmul(u[2], base);
    base = ccubeTrig(base, w);
    for (u32 i = 3; i < n; ++i) {
      u[i] = cmul(u[i], base);
      base = cmul(base, w);
    }
  }
#else
#error CLEAN must be 0 or 1
#endif
}


// Preload trig values for the first partial tabMul.  We load the sine/cosine values early so that F64 ops can hide the read latency.
void preload_tabMul_trig(u32 WG, Trig trig, T *preloads, u32 f, u32 me) {
  u32 old_width_trigs = WG*8;
  TrigSingle trig1 = (TrigSingle) (trig + old_width_trigs);

  // Read 7 lines of sine/cosine values for the first fft8.  Read six of the lines as pairs as AMD likes T2 global memory reads
  for (u32 i = 1; i < 7; i += 2) {
    Trig trig2 = (Trig) (trig1 + (i-1)*WG);
    T2 sine_over_cosines = trig2[me];
    preloads[i-1] = sine_over_cosines.x;
    preloads[i] = sine_over_cosines.y;
  }
  // Read 7th line
  preloads[6] = trig1[6*WG + me];
}

// Partial complex-multiply that delays the mul-by-cosine so it can be part of an FMA.
// We're trying to calculate u * U2(cosine,sine).
// real = (u.x - u.y*sine_over_cosine) * cosine
// imag = (u.x*sine_over_cosine + u.y) * cosine
T2 partial_cmul(T2 u, T sine_over_cosine) {
  return U2(fma(-u.y, sine_over_cosine, u.x), fma(u.x, sine_over_cosine, u.y));
}

// Do a partial tabMul.  Save the mul-by-cosine for later FMA instructions.
void partial_tabMul(u32 WG, local T2 *lds, Trig trig, T *preloads, T2 *u, u32 f, u32 me) {
  local T *lds1 = (local T *) lds;
  u32 old_width_trigs = WG*8;
  TrigSingle trig1 = (TrigSingle) (trig + old_width_trigs);
  trig1 += 8*WG;                                // Skip past sine_over_cosine values

  // Use LDS memory to distribute preloaded trig values.
  if (f > 1) {
    lds1[me] = preloads[4];     // Preloaded sine/cosine values
    lds1[WG+me] = preloads[5];  // Preloaded cosine values
    bar(WG);
  }

  // Apply sine/cosines
  for (u32 i = 1; i < 8; ++i) {
    T sine_over_cosine;
    if (f == 1) sine_over_cosine = preloads[i-1];
    else sine_over_cosine = lds1[i*8 + me/8];
    u[i] = partial_cmul(u[i], sine_over_cosine);
  }

  // Preload cosines for finishing first tabMul (done after using up preloaded sine/cosine values).  Hopefully, shufl will hide the latency.
  if (f == 1) {
    // Read pairs of lines to make AMD happy with T2 global memory loads
    for (u32 i = 0; i < 8; i += 2) {
      Trig trig2 = (Trig) (trig1 + i*WG);
      T2 cosines = trig2[me];
      preloads[i] = cosines.x;
      preloads[i+1] = cosines.y;
    }
  }
  else {
    // Load cosine4, cosine5/cosine1, cosine6/cosine2, cosine7/cosine3, cosine2, cosine3/cosine1, cosine1
    // Load them in the order they will be used, though it probably won't matter.
    preloads[1] = lds1[WG + 1*8 + me/8];
    preloads[4] = lds1[WG + 4*8 + me/8];
    preloads[5] = lds1[WG + 5*8 + me/8];
    preloads[6] = lds1[WG + 6*8 + me/8];
    preloads[7] = lds1[WG + 7*8 + me/8];
    preloads[2] = lds1[WG + 2*8 + me/8];
    preloads[3] = lds1[WG + 3*8 + me/8];
    bar(WG);
  }
}

// Copy of macros from fft8 with FMAs added
#define X2_via_FMA(a, c, b) { T2 t = a; a = fma(c, b, t); b = fma(-c, b, t); }

// Finish off a partial tabMul while doing next fft8 making more use of FMA.
void finish_tabMul_fft8(u32 WG, local T2 *lds, Trig trig, T *preloads, T2 *u, u32 f, u32 me, u32 save_one_more_mul) {
  local T *lds1 = (local T *) lds;
  u32 old_width_trigs = WG*8;
  TrigSingle trig1 = (TrigSingle) (trig + old_width_trigs);

  //
  // Mimic a traditional fft8 but use FMA instructions to apply the cosine multiplies.
  //

  // Apply cosine0 to u[0]
  if (f == 1) u[0] = u[0] * preloads[0];

 if (save_one_more_mul) {   // This should always be the best option.  ROCm optimizer is doing something weird in new_fft_WIDTH case.

  // Apply cosine4, cosine5/cosine1, cosine6/cosine2, cosine7/cosine3 to u[4] through u[7] using FMA
  X2_via_FMA(u[0], preloads[4], u[4]);
  X2_via_FMA(u[1], preloads[5], u[5]);  u[5] = mul_t8_delayed(u[5]);
  X2_via_FMA(u[2], preloads[6], u[6]);  u[6] = mul_t4(u[6]);
  X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);

  // Preload one line of sine/cosines and one line of cosines for second tabMul.  We'll later broadcast these values as needed using LDS.
  if (f == 1) {
    preloads[4] = trig1[7*WG + me];             // Sine/cosines for second tabMul
    preloads[5] = trig1[8*WG + 8*WG + me];      // Cosines for second tabMul
  }

  // Do the fft4Core and fft4CoreSpecial applying cosine2, cosine3/cosine1
  X2_via_FMA(u[0], preloads[2], u[2]);
  X2_via_FMA(u[4], preloads[2], u[6]);
  X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);
  X2_via_FMA(u[5], preloads[3], u[7]);  u[7] = mul_t4(u[7]);

  // Do last level of fft8 applying cosine1
//TODO: Save this MUL by SQRT(1/2) by pre-computing cosine1*SQRTHALF
  T cosine1_SQRT1_2 = preloads[1] * M_SQRT1_2;
  X2_via_FMA(u[0], preloads[1], u[1]);
  X2_via_FMA(u[2], preloads[1], u[3]);
  X2_via_FMA(u[4], cosine1_SQRT1_2, u[5]);
  X2_via_FMA(u[6], cosine1_SQRT1_2, u[7]);

 } else {

  // Apply cosine to u[1]
  u[1] = u[1] * preloads[1];

  // Apply cosine4, cosine5, cosine6/cosine2, cosine7/cosine3 to u[4] through u[7] using FMA
  X2_via_FMA(u[0], preloads[4], u[4]);
  X2_via_FMA(u[1], preloads[5], u[5]);  u[5] = mul_t8_delayed(u[5]);
  X2_via_FMA(u[2], preloads[6], u[6]);  u[6] = mul_t4(u[6]);
  X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);

  // Preload one line of sine/cosines and one line of cosines for second tabMul.  We'll later broadcast these values as needed using LDS.
  if (f == 1) {
    preloads[4] = trig1[7*WG + me];             // Sine/cosines for second tabMul
    preloads[5] = trig1[8*WG + 8*WG + me];      // Cosines for second tabMul
  }

  // Do the fft4Core and fft4CoreSpecial applying cosine2, cosine3
  X2_via_FMA(u[0], preloads[2], u[2]);
  X2_via_FMA(u[4], preloads[2], u[6]);
  X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);
  X2_via_FMA(u[5], preloads[3], u[7]);  u[7] = mul_t4(u[7]);

  // Do last level of fft8
  X2(u[0], u[1]);
  X2(u[2], u[3]);
  X2_apply_delay(u[4], u[5]);
  X2_apply_delay(u[6], u[7]);

 }

  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

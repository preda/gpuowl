// Copyright (C) Mihai Preda

#include "fft4.cl"
#include "fft8.cl"
#include "trig.cl"
// #include "math.cl"

void chainMul4(T2 *u, T2 w) {
  u[1] = cmul(u[1], w);

  T2 base = csqTrig(w);
  u[2] = cmul(u[2], base);

  double a = 2 * base.y;
  base = U2(fma(a, -w.y, w.x), fma(a, w.x, -w.y));
  u[3] = cmul(u[3], base);
}

#if 1
// This version of chainMul8 tries to minimize roundoff error even if more F64 ops are used.
// Trial and error looking at Z values on a WIDTH=512 FFT was used to determine when to switch from fancy to non-fancy powers of w.
void chainMul8(T2 *u, T2 w, u32 tailSquareBcast) {
  u[1] = cmulFancy(u[1], w);

  T2 w2;
  // Rocm optimizer behaves weirdly. Using multiple mul2s instead of one mul2 in csqTrigFancy makes double-wide single-kernel tailSquare inexplicably slower
  if (!tailSquareBcast) {
    w2 = csqTrigFancy(w);
  } else {
    w2 = U2(mulminus2(w.y) * w.y, mul2(fma(w.x, w.y, w.y)));
  }
  u[2] = cmulFancy(u[2], w2);

  T2 w3;
  // Rocm optimizer behaves weirdly yet again. Using mul2 instead of 2.0* makes double-wide single-kernel tailSquare inexplicably slower
  // even though it is one fewer F64 op.
  if (!tailSquareBcast) {
    w3 = ccubeTrigFancy(w2, w);
  } else {
    double a = 2*w2.y;
    w3 = U2(fma(a, -w.y, w.x), fma(a, w.x, a - w.y));
  }
  u[3] = cmulFancy(u[3], w3);

  w3.x += 1;
  T2 base = cmulFancy (w3, w);
  for (int i = 4; i < 8; ++i) {
    u[i] = cmul(u[i], base);
    base = cmulFancy(base, w);
  }
}

#else
// This version of chainMul8 minimizes F64 ops even if that increases roundoff error.
// This version is faster on a Radeon 7 with worse roundoff in :0 fft spec.  The :2 fft spec is even faster with no roundoff penalty.
// This version is the same speed on a TitanV due to its great F64 throughput.
// This version is slower on R7Pro due to a rocm optimizer issue in double-wide single-kernel tailSquare using BCAST.  I could not find a work-around.
// Other GPUs?  This version might be useful.
void chainMul8(T2 *u, T2 w, u32 tailSquareBcast) {
  u[1] = cmulFancy(u[1], w);

  T2 w2 = csqTrigFancy(w);
  u[2] = cmulFancy(u[2], w2);

  T2 w3 = ccubeTrigDefancy(w2, w);
  u[3] = cmul(u[3], w3);

  T2 w4 = csqTrigDefancy(w2);
  u[4] = cmul(u[4], w4);

  T2 w6 = csqTrig(w3);
  T2 w5, w7; cmul_a_by_fancyb_and_conjfancyb(&w7, &w5, w6, w);
  u[5] = cmul(u[5], w5);
  u[6] = cmul(u[6], w6);
  u[7] = cmul(u[7], w7);
}
#endif

void chainMul(u32 len, T2 *u, T2 w, u32 tailSquareBcast) {
  // Do a length 4 chain mul, w must not be in Fancy format
  if (len == 4) chainMul4(u, w);
  // Do a length 8 chain mul, w must be in Fancy format
  if (len == 8) chainMul8(u, w, tailSquareBcast);
}


#if AMDGPU

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

// Same as shufl but use ints instead of doubles to reduce LDS memory requirements.
// Lower LDS requirements should let the optimizer use fewer VGPRs and increase occupancy for WIDTHs >= 1024.
// Alas, the increased occupancy does not offset extra code needed for shufl_int (the assembly
// code generated is not pretty).  This might not be true for nVidia or future ROCm optimizers.
void shufl_int(u32 WG, local T2 *lds2, T2 *u, u32 n, u32 f) {
  u32 me = get_local_id(0);
  local int* lds = (local int*) lds2;

  u32 mask = f - 1;
  assert((mask & (mask + 1)) == 0);
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = as_int4(u[i]).x; }
  bar();
  for (u32 i = 0; i < n; ++i) { int4 tmp = as_int4(u[i]); tmp.x = lds[i * WG + me]; u[i] = as_double2(tmp); }
  bar();
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = as_int4(u[i]).y; }
  bar();
  for (u32 i = 0; i < n; ++i) { int4 tmp = as_int4(u[i]); tmp.y = lds[i * WG + me]; u[i] = as_double2(tmp); }
  bar();
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = as_int4(u[i]).z; }
  bar();
  for (u32 i = 0; i < n; ++i) { int4 tmp = as_int4(u[i]); tmp.z = lds[i * WG + me]; u[i] = as_double2(tmp); }
  bar();
  for (u32 i = 0; i < n; ++i) { lds[i * f + (me & ~mask) * n + (me & mask)] = as_int4(u[i]).w; }
  bar();
  for (u32 i = 0; i < n; ++i) { int4 tmp = as_int4(u[i]); tmp.w = lds[i * WG + me]; u[i] = as_double2(tmp); }
  bar();   // I'm not sure why this barrier call is needed
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

void tabMul(u32 WG, Trig trig, T2 *u, u32 n, u32 f, u32 me, bool chainmul) {
#if 0
  u32 p = me / f * f;
#else
  u32 p = me & ~(f - 1);
#endif

// Compute trigs from scratch every time.  This can't possibly be a good idea on any GPUs.
#if 0
  T2 w = slowTrig_N(ND / n / WG * p, ND / n);
  T2 base = w;
  for (int i = 1; i < n; ++i) {
    u[i] = cmul(u[i], w);
    w = cmul(w, base);
  }
  return;
#endif

// This code uses chained complex multiplies which could be faster on GPUs with great DP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice this is more accurate (at least when n==8) than reading precomputed
// values from memory.  Perhaps chained Fancy muls are the reason (or was resolved when the algorithm to precompute trig values changed).

  if (chainmul) {
    T2 w = trig[p];
    chainMul (n, u, w, 0);
    return;
  }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.
// Radeon VII loves this case, it is faster than the chainmul case.  nVidia Titan V hates this case.

  if (!chainmul) {
    T2 w = trig[p];

    if (n >= 8) {
      u[1] = cmulFancy(u[1], w);
    } else {
      u[1] = cmul(u[1], w);
    }

    for (u32 i = 2; i < n; ++i) {
      T2 base = trig[(i-1)*WG + p];
      u[i] = cmul(u[i], base);
    }
    return;
  }
}


//************************************************************************************
// New fft WIDTH and HEIGHT macros to support radix-4 FFTs with more FMA instructions
//************************************************************************************

// Partial complex-multiply that delays the mul-by-cosine so it can be part of an FMA.
// We're trying to calculate u * U2(cosine,sine).
// real = (u.x - u.y*sine_over_cosine) * cosine
// imag = (u.x*sine_over_cosine + u.y) * cosine
T2 partial_cmul(T2 u, T sine_over_cosine) {
  return U2(fma(-u.y, sine_over_cosine, u.x), fma(u.x, sine_over_cosine, u.y));
}

// Copy of macro from fft4 and fft8 with FMAs added
#define X2_via_FMA(a, c, b) { T2 t = a; a = fma(c, b, t); b = fma(-c, b, t); }

// Preload trig values for the first partial tabMul.  We load the sine/cosine values early so that F64 ops can hide the read latency.
void preload_tabMul4_trig(u32 WG, Trig trig, T *preloads, u32 f, u32 me) {
  TrigSingle trig1 = (TrigSingle) trig;

  // Read 3 lines of sine/cosine values for the first fft4.  Read two of the lines as a pair as AMD likes T2 global memory reads
  Trig trig2 = (Trig) trig1;
  T2 sine_over_cosines = trig2[me];
  preloads[0] = sine_over_cosines.x;
  preloads[1] = sine_over_cosines.y;
  // Read 3rd line
  preloads[2] = trig1[2*WG + me];
}

// Do a partial tabMul.  Save the mul-by-cosine for later FMA instructions.
void partial_tabMul4(u32 WG, local T2 *lds, Trig trig, T *preloads, T2 *u, u32 f, u32 me) {
  local T *lds1 = (local T *) lds;
  TrigSingle trig1 = (TrigSingle) trig;
  trig1 += 4*WG;                // Skip past sine_over_cosine values

  // Use LDS memory to distribute preloaded trig values.
  if (f > 1) {
    lds1[me] = preloads[4];     // Preloaded sine/cosine values
    lds1[WG+me] = preloads[5];  // Preloaded cosine values
    bar(WG);
  }

  // Apply sine/cosines
  for (u32 i = 1; i < 4; ++i) {
    T sine_over_cosine;
    if (f == 1) sine_over_cosine = preloads[i-1];
    else sine_over_cosine = lds1[i*(WG/4) + (me/f)*(f/4)];
    u[i] = partial_cmul(u[i], sine_over_cosine);
  }

  // Preload cosines for finishing first tabMul (done after using up preloaded sine/cosine values).  Hopefully, shufl will hide the latency.
  if (f == 1) {
    // Read pairs of lines to make AMD happy with T2 global memory loads
    for (u32 i = 0; i < 4; i += 2) {
      Trig trig2 = (Trig) (trig1 + i*WG);
      T2 cosines = trig2[me];
      preloads[i] = cosines.x;
      preloads[i+1] = cosines.y;
    }
  }
  else {
    // Load cosine1, cosine2, cosine3/cosine1
    if (f < WG/4) preloads[0] = lds1[WG + ((me/f) & 3) * WG/4 + (0 * WG + me)/(4*f) * f/4];
    preloads[2] = lds1[WG + ((me/f) & 3) * WG/4 + (2 * WG + me)/(4*f) * f/4];
    preloads[3] = lds1[WG + ((me/f) & 3) * WG/4 + (3 * WG + me)/(4*f) * f/4];
    preloads[1] = lds1[WG + ((me/f) & 3) * WG/4 + (1 * WG + me)/(4*f) * f/4];
    bar(WG);
  }
}

// Finish off a partial tabMul while doing next fft4 making more use of FMA.
void finish_tabMul4_fft4(u32 WG, local T2 *lds, Trig trig, T *preloads, T2 *u, u32 f, u32 me, u32 save_one_more_mul) {
  local T *lds1 = (local T *) lds;
  TrigSingle trig1 = (TrigSingle) trig;

  //
  // Mimic a traditional fft4 but use FMA instructions to apply the cosine multiplies.
  //

  // Apply cosine0 to u[0]
  if (f < WG/4) u[0] = u[0] * preloads[0];

  // Apply cosine2, cosine3/cosine1 to u[2] and u[3] using FMA
  X2_via_FMA(u[0], preloads[2], u[2]);
  X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);

  // Preload one line of sine/cosines and one line of cosines for later tabMuls.  We'll later broadcast these values as needed using LDS.
  if (f == 1) {
    preloads[4] = trig1[3*WG + me];             // Sine/cosines for later tabMuls
    preloads[5] = trig1[4*WG + 4*WG + me];      // Cosines for later tabMuls
  }

  // Do the last level of fft4 applying cosine1
  X2_via_FMA(u[0], preloads[1], u[1]);
  X2_via_FMA(u[2], preloads[1], u[3]);

  // revbin [0, 2, 1, 3] undo
  SWAP(u[1], u[2]);
}

//************************************************************************************
// New fft WIDTH and HEIGHT macros to support radix-8 FFTs with more FMA instructions
//************************************************************************************

// Preload trig values for the first partial tabMul.  We load the sine/cosine values early so that F64 ops can hide the read latency.
void preload_tabMul8_trig(u32 WG, Trig trig, T *preloads, u32 f, u32 me) {
  TrigSingle trig1 = (TrigSingle) trig;

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

// Do a partial tabMul.  Save the mul-by-cosine for later FMA instructions.
void partial_tabMul8(u32 WG, local T2 *lds, Trig trig, T *preloads, T2 *u, u32 f, u32 me) {
  local T *lds1 = (local T *) lds;
  TrigSingle trig1 = (TrigSingle) trig;
  trig1 += 8*WG;                // Skip past sine_over_cosine values

  // Use LDS memory to distribute preloaded trig values.
  if (f > 1) {
    lds1[me] = preloads[8];     // Preloaded sine/cosine values
    lds1[WG+me] = preloads[9];  // Preloaded cosine values
    bar(WG);
  }

  // Apply sine/cosines
  for (u32 i = 1; i < 8; ++i) {
    T sine_over_cosine;
    if (f == 1) sine_over_cosine = preloads[i-1];
    else sine_over_cosine = lds1[i*(WG/8) + (me/f)*(f/8)];
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
    if (f < WG/8) preloads[0] = lds1[WG + ((me/f) & 7) * WG/8 + (0 * WG + me)/(8*f) * f/8];
    preloads[1] = lds1[WG + ((me/f) & 7) * WG/8 + (1 * WG + me)/(8*f) * f/8];
    preloads[4] = lds1[WG + ((me/f) & 7) * WG/8 + (4 * WG + me)/(8*f) * f/8];
    preloads[5] = lds1[WG + ((me/f) & 7) * WG/8 + (5 * WG + me)/(8*f) * f/8];
    preloads[6] = lds1[WG + ((me/f) & 7) * WG/8 + (6 * WG + me)/(8*f) * f/8];
    preloads[7] = lds1[WG + ((me/f) & 7) * WG/8 + (7 * WG + me)/(8*f) * f/8];
    preloads[2] = lds1[WG + ((me/f) & 7) * WG/8 + (2 * WG + me)/(8*f) * f/8];
    preloads[3] = lds1[WG + ((me/f) & 7) * WG/8 + (3 * WG + me)/(8*f) * f/8];
    bar(WG);
  }
}

// Finish off a partial tabMul while doing next fft8 making more use of FMA.
void finish_tabMul8_fft8(u32 WG, local T2 *lds, Trig trig, T *preloads, T2 *u, u32 f, u32 me, u32 save_one_more_mul) {
  local T *lds1 = (local T *) lds;
  TrigSingle trig1 = (TrigSingle) trig;

  //
  // Mimic a traditional fft8 but use FMA instructions to apply the cosine multiplies.
  //

  // Apply cosine0 to u[0]
  if (f < WG/8) u[0] = u[0] * preloads[0];

  if (save_one_more_mul) {   // This should always be the best option.  ROCm optimizer is doing something weird in new_fft_WIDTH case.

    // Apply cosine4, cosine5/cosine1, cosine6/cosine2, cosine7/cosine3 to u[4] through u[7] using FMA
    X2_via_FMA(u[0], preloads[4], u[4]);
    X2_via_FMA(u[1], preloads[5], u[5]);  u[5] = mul_t8_delayed(u[5]);
    X2_via_FMA(u[2], preloads[6], u[6]);  u[6] = mul_t4(u[6]);
    X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);

    // Preload one line of sine/cosines and one line of cosines for second tabMul.  We'll later broadcast these values as needed using LDS.
    if (f == 1) {
      preloads[8] = trig1[7*WG + me];             // Sine/cosines for second tabMul
      preloads[9] = trig1[8*WG + 8*WG + me];      // Cosines for second tabMul
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
      preloads[8] = trig1[7*WG + me];             // Sine/cosines for second tabMul
      preloads[9] = trig1[8*WG + 8*WG + me];      // Cosines for second tabMul
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

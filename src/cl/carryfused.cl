// Copyright (C) Mihai Preda

#include "carryutil.cl"
#include "weight.cl"
#include "fftwidth.cl"
#include "middle.cl"

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W) carryFused(P(T2) out, CP(T2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
                       CP(u32) bits, BigTab THREAD_WEIGHTS, P(uint) bufROE) {
  local T2 lds[WIDTH / 2];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
  u32 line = gr % H;

  T2 u[NW];
  
  readCarryFusedLine(in, u, line);

  // Split 32 bits into NW groups of 2 bits.
#define GPW (16 / NW)
  u32 b = bits[(G_W * line + me) / GPW] >> (me % GPW * (2 * NW));
#undef GPW
  
  fft_WIDTH(lds, u, smallTrig);

  Word2 wu[NW];
  T2 weights = fancyMul(THREAD_WEIGHTS[me], THREAD_WEIGHTS[G_W + line]);

#if MUL3
  P(i64) carryShuttlePtr = (P(i64)) carryShuttle;
  i64 carry[NW+1];
#else
  P(CFcarry) carryShuttlePtr = (P(CFcarry)) carryShuttle;
  CFcarry carry[NW+1];
#endif

  float roundMax = 0;
  float carryMax = 0;
  
  // Apply the inverse weights

  T invBase = optionalDouble(weights.x);
  
  for (u32 i = 0; i < NW; ++i) {
    T invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)));
    T invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP));

    u[i] = conjugate(u[i]) * U2(invWeight1, invWeight2);
  }

  // Generate our output carries
  for (i32 i = 0; i < NW; ++i) {
#if MUL3
    wu[i] = carryPairMul(u[i], &carry[i], test(b, 2 * i), test(b, 2 * i + 1), 0, &roundMax, &carryMax);    
#else
    wu[i] = carryPair(u[i], &carry[i], test(b, 2 * i), test(b, 2 * i + 1),
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, &roundMax, &carryMax);
#endif
  }

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries. Only groups 0 to H-1 need to write carries out.
  // Group H is a duplicate of group 0 (producing the same results) so we don't care about group H writing out,
  // but it's fine either way.
  if (gr < H) { for (i32 i = 0; i < NW; ++i) { carryShuttlePtr[gr * WIDTH + me * NW + i] = carry[i]; } }

#if OLD_FENCE

  if (gr < H) {
    work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    if (me == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
  }
  if (gr == 0) { return; }
  if (me == 0) { while(!atomic_load((atomic_uint *) &ready[gr - 1])); }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);

#else

  if (gr < H) {
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (me % WAVEFRONT == 0) { atomic_store((atomic_uint *) &ready[gr * (G_W / WAVEFRONT) + me / WAVEFRONT], 1); }
  }
  if (gr == 0) { return; }
  if (me % WAVEFRONT == 0) {
    while(!atomic_load((atomic_uint *) &ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT]));
  }
  read_mem_fence(CLK_GLOBAL_MEM_FENCE);

#endif

  // Read from the carryShuttle carries produced by the previous WIDTH row.  Rotate carries from the last WIDTH row.
  // The new carry layout lets the compiler generate global_load_dwordx4 instructions.
  if (gr < H) {
    for (i32 i = 0; i < NW; ++i) {
      carry[i] = carryShuttlePtr[(gr - 1) * WIDTH + me * NW + i];
    }
  } else {

#if !OLD_FENCE
    // For gr==H we need the barrier since the carry reading is shifted, thus the per-wavefront trick does not apply.
    work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
#endif

    for (i32 i = 0; i < NW; ++i) {
      carry[i] = carryShuttlePtr[(gr - 1) * WIDTH + (me + G_W - 1) % G_W * NW + i /* ((me!=0) + NW - 1 + i) % NW*/];
    }

    if (me == 0) {
      // Tcarry tmp = carry[NW - 1];

      carry[NW] = carry[NW-1];
      for (i32 i = NW-1; i; --i) { carry[i] = carry[i-1]; }
      carry[0] = carry[NW];
    }
  }

  // Apply each 32 or 64 bit carry to the 2 words
  for (i32 i = 0; i < NW; ++i) {
    wu[i] = carryFinal(wu[i], carry[i], test(b, 2 * i));
  }
  
  T base = optionalHalve(weights.y);
  
  for (u32 i = 0; i < NW; ++i) {
    T weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    T weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP));
    u[i] = U2(wu[i].x, wu[i].y) * U2(weight1, weight2);
  }

  bar();

  fft_WIDTH(lds, u, smallTrig);
  write(G_W, NW, u, out, WIDTH * line);

  // Clear carry ready flag for next iteration
#if OLD_FENCE
  if (me == 0) ready[gr - 1] = 0;
#else
  if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
}

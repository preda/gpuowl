// Copyright (C) Mihai Preda

#include "carryutil.cl"
#include "weight.cl"
#include "fftwidth.cl"
#include "middle.cl"

void spin() {
#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_s_sleep)
  __builtin_amdgcn_s_sleep(0);
#elif HAS_ASM
  __asm("s_sleep 0");
#else
  // nothing: just spin
  // on Nvidia: see if there's some brief sleep function
#endif
}

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W) carryFused(P(T2) out, CP(T2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
		       CP(u32) bits, ConstBigTab CONST_THREAD_WEIGHTS, BigTab THREAD_WEIGHTS, P(uint) bufROE) {

#if 0   // fft_WIDTH uses shufl_int instead of shufl
  local T2 lds[WIDTH / 4];
#else
  local T2 lds[WIDTH / 2];
#endif
  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
  u32 line = gr % H;

  T2 u[NW];

#if HAS_ASM
  __asm("s_setprio 3");
#endif

  readCarryFusedLine(in, u, line);

// Split 32 bits into NW groups of 2 bits.  See later for different way to do this.
#if !BIGLIT
#define GPW (16 / NW)
  u32 b = bits[(G_W * line + me) / GPW] >> (me % GPW * (2 * NW));
#undef GPW
#endif

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
//  fft_WIDTH(lds + (get_group_id(0) / 131072), u, smallTrig + (get_group_id(0) / 131072));

// A temporary hack until we figure out which combinations we want to finally offer:
// UNROLL_W=0: old fft_WIDTH, no loop unrolling
// UNROLL_W=1: old fft_WIDTH, loop unrolling
// UNROLL_W=2: old fft_WIDTH, loop unrolling with "hidden zero" hack to thwart rocm optimizer.  I'm seeing this as best R7Pro option.
// UNROLL_W=3: new fft_WIDTH if applicable, hidden zero hack.  Slightly better on Radeon VII -- more study needed as to why results weren't better.
// UNROLL_W=4: new fft_WIDTH if applicable, no hidden zero hack.  Best on Titan V.
#if UNROLL_W == 2 || UNROLL_W == 3
  new_fft_WIDTH1(lds + (get_group_id(0) / 131072), u, smallTrig + (get_group_id(0) / 131072));
#else
  new_fft_WIDTH1(lds, u, smallTrig);
#endif

  Word2 wu[NW];
#if AMDGPU
  T2 weights = fancyMul(THREAD_WEIGHTS[me], THREAD_WEIGHTS[G_W + line]);
#else
  T2 weights = fancyMul(CONST_THREAD_WEIGHTS[me], THREAD_WEIGHTS[G_W + line]);            // On nVidia, don't pollute the constant cache with line weights
#endif

#if MUL3
  P(i64) carryShuttlePtr = (P(i64)) carryShuttle;
  i64 carry[NW+1];
#else
  P(CFcarry) carryShuttlePtr = (P(CFcarry)) carryShuttle;
  CFcarry carry[NW+1];
#endif

#if AMDGPU
#define CarryShuttleAccess(me,i)        ((me) * NW + (i))                       // Generates denser global_load_dwordx4 instructions
//#define CarryShuttleAccess(me,i)      ((me) * 4 + (i)%4 + (i)/4 * 4*G_W)      // Also generates global_load_dwordx4 instructions and unit stride when NW=8
#else
#define CarryShuttleAccess(me,i)        ((me) + (i) * G_W)                      // nVidia likes this better unit stride better
#endif

  float roundMax = 0;
  float carryMax = 0;
  
  // Apply the inverse weights

  T invBase = optionalDouble(weights.x);
  
  for (u32 i = 0; i < NW; ++i) {
    T invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)));
    T invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP));

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    * ((long2*) &u[i]) = weightAndCarry(conjugate(u[i]), U2(invWeight1, invWeight2),
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, &roundMax);
  }

  // On Titan V it is faster to derive the big vs. little flags from the fractional number of bits in each FFT word rather read the flags from memory.
  // On Radeon VII this code is about the same speed.  Not sure which is better on other GPUs.
#if BIGLIT
  // Calculate the most significant 32-bits of FRAC_BPW * the index of the FFT word.  Also add FRAC_BPW_HI to test first biglit flag.
  u32 fft_word_index = (me * H + line) * 2;
  u32 frac_bits = fft_word_index * FRAC_BPW_HI + mad_hi (fft_word_index, FRAC_BPW_LO, FRAC_BPW_HI);
  u32 tmp = frac_bits;
#endif

  // Generate our output carries
  for (i32 i = 0; i < NW; ++i) {
#if BIGLIT
    bool biglit0 = tmp <= FRAC_BPW_HI; tmp += FRAC_BPW_HI;
    bool biglit1 = tmp <= FRAC_BPW_HI; tmp += FRAC_BITS_BIGSTEP - FRAC_BPW_HI;
#else
    bool biglit0 = test(b, 2 * i);
    bool biglit1 = test(b, 2 * i + 1);
#endif

    // Propagate carries through two words.  Generate the output carry.
    wu[i] = carryPair(*(long2*)&u[i], &carry[i], biglit0, biglit1, &carryMax);
  }

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries. Only groups 0 to H-1 need to write carries out.
  // Group H is a duplicate of group 0 (producing the same results) so we don't care about group H writing out,
  // but it's fine either way.
  if (gr < H) { for (i32 i = 0; i < NW; ++i) { carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(me, i)] = carry[i]; } }

#if OLD_FENCE

  if (gr < H) {
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar();
    
    if (me == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
  }
  if (gr == 0) { return; }
  if (me == 0) { do { spin(); } while(!atomic_load((atomic_uint *) &ready[gr - 1])); }
  // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
  bar();
  read_mem_fence(CLK_GLOBAL_MEM_FENCE);

  // Clear carry ready flag for next iteration
  if (me == 0) ready[gr - 1] = 0;

#else

  if (gr < H) {
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (me % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + me / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
  }
  if (gr == 0) { return; }
#if HAS_ASM
  __asm("s_setprio 0");
#endif
  u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
  if (me % WAVEFRONT == 0) {
    do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
  }
#if HAS_ASM
  __asm("s_setprio 1");
#endif
  mem_fence(CLK_GLOBAL_MEM_FENCE);

  // Clear carry ready flag for next iteration
  if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif

  // Read from the carryShuttle carries produced by the previous WIDTH row.  Rotate carries from the last WIDTH row.
  // The new carry layout lets the compiler generate global_load_dwordx4 instructions.
  if (gr < H) {
    for (i32 i = 0; i < NW; ++i) {
      carry[i] = carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess(me, i)];
    }
  } else {

#if !OLD_FENCE
    // For gr==H we need the barrier since the carry reading is shifted, thus the per-wavefront trick does not apply.
    bar();
#endif

    for (i32 i = 0; i < NW; ++i) {
      carry[i] = carryShuttlePtr[(gr - 1) * WIDTH + CarryShuttleAccess((me + G_W - 1) % G_W, i) /* ((me!=0) + NW - 1 + i) % NW*/];
    }

    if (me == 0) {
      carry[NW] = carry[NW-1];
      for (i32 i = NW-1; i; --i) { carry[i] = carry[i-1]; }
      carry[0] = carry[NW];
    }
  }

  // Apply each 32 or 64 bit carry to the 2 words
  for (i32 i = 0; i < NW; ++i) {
#if BIGLIT
    bool biglit0 = frac_bits <= FRAC_BPW_HI; frac_bits += FRAC_BITS_BIGSTEP;
#else
    bool biglit0 = test(b, 2 * i);
#endif
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
  }
  
  T base = optionalHalve(weights.y);
  
  for (u32 i = 0; i < NW; ++i) {
    T weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    T weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP));
    u[i] = U2(wu[i].x, wu[i].y) * U2(weight1, weight2);
  }

  bar();

//  fft_WIDTH(lds, u, smallTrig);
  new_fft_WIDTH2(lds, u, smallTrig);

  writeCarryFusedLine(u, out, line);
}

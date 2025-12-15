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

#if FFT_TYPE == FFT64

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W) carryFused(P(T2) out, CP(T2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
                       CP(u32) bits, ConstBigTab CONST_THREAD_WEIGHTS, BigTab THREAD_WEIGHTS, P(uint) bufROE) {

#if 0   // fft_WIDTH uses shufl_int instead of shufl
  local T2 lds[WIDTH / 4];
#else
  local T2 lds[WIDTH / 2];
#endif

  T2 u[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
  u32 line = gr % H;

#if HAS_ASM
  __asm("s_setprio 3");
#endif

  readCarryFusedLine(in, u, line);

// Split 32 bits into NW groups of 2 bits.  See later for different way to do this.
#if !BIGLIT
#define GPW (16 / NW)
  u32 b = NTLOAD(bits[(G_W * line + me) / GPW]) >> (me % GPW * (2 * NW));
#undef GPW
#endif

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
#if ZEROHACK_W
  u32 zerohack = get_group_id(0) / 131072;
  new_fft_WIDTH1(lds + zerohack, u, smallTrig + zerohack);
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
#define CarryShuttleAccess(me,i)        ((me) + (i) * G_W)                      // nVidia likes this unit stride better
#endif

  float roundMax = 0;
  float carryMax = 0;

  // On Titan V it is faster to derive the big vs. little flags from the fractional number of bits in each FFT word rather than read the flags from memory.
  // On Radeon VII this code is about the same speed.  Not sure which is better on other GPUs.
#if BIGLIT
  // Calculate the most significant 32-bits of FRAC_BPW * the word index.  Also add FRAC_BPW_HI to test first biglit flag.
  u32 word_index = (me * H + line) * 2;
  u32 frac_bits = word_index * FRAC_BPW_HI + mad_hi (word_index, FRAC_BPW_LO, FRAC_BPW_HI);
  const u32 frac_bits_bigstep = ((G_W * H * 2) * FRAC_BPW_HI + (u32)(((u64)(G_W * H * 2) * FRAC_BPW_LO) >> 32));
#endif

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  T invBase = optionalDouble(weights.x);

  for (u32 i = 0; i < NW; ++i) {
    T invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)));
    T invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP));

    // Generate big-word/little-word flags
#if BIGLIT
    bool biglit0 = frac_bits + i * frac_bits_bigstep <= FRAC_BPW_HI;
    bool biglit1 = frac_bits + i * frac_bits_bigstep >= -FRAC_BPW_HI;   // Same as frac_bits + i * frac_bits_bigstep + FRAC_BPW_HI <= FRAC_BPW_HI;
#else
    bool biglit0 = test(b, 2 * i);
    bool biglit1 = test(b, 2 * i + 1);
#endif

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(u[i]), U2(invWeight1, invWeight2),
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);
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

  // Tell next line that its carries are ready
  if (gr < H) {
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar();
    if (me == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (me % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + me / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Line zero will be redone when gr == H
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Calculate inverse weights
  T base = optionalHalve(weights.y);
  for (u32 i = 0; i < NW; ++i) {
    T weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    T weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP));
    u[i] = U2(weight1, weight2);
  }

  // Wait until our carries are ready
#if OLD_FENCE
  if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
  // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
  bar();
  read_mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me == 0) ready[gr - 1] = 0;
#else
  u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
  if (me % WAVEFRONT == 0) {
    do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
  __asm("s_setprio 1");
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
    bool biglit0 = frac_bits + i * frac_bits_bigstep <= FRAC_BPW_HI;
#else
    bool biglit0 = test(b, 2 * i);
#endif
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    u[i] = U2(u[i].x * wu[i].x, u[i].y * wu[i].y);
  }

  bar();

//  fft_WIDTH(lds, u, smallTrig);
  new_fft_WIDTH2(lds, u, smallTrig);

  writeCarryFusedLine(u, out, line);
}


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#elif FFT_TYPE == FFT32

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W) carryFused(P(F2) out, CP(F2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, TrigFP32 smallTrig,
                       CP(u32) bits, ConstBigTabFP32 CONST_THREAD_WEIGHTS, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {

#if 0   // fft_WIDTH uses shufl_int instead of shufl
  local F2 lds[WIDTH / 4];
#else
  local F2 lds[WIDTH / 2];
#endif

  F2 u[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
  u32 line = gr % H;

#if HAS_ASM
  __asm("s_setprio 3");
#endif

  readCarryFusedLine(in, u, line);

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
#if ZEROHACK_W
  u32 zerohack = get_group_id(0) / 131072;
  new_fft_WIDTH1(lds + zerohack, u, smallTrig + zerohack);
#else
  new_fft_WIDTH1(lds, u, smallTrig);
#endif

  Word2 wu[NW];
#if AMDGPU
  F2 weights = fancyMul(THREAD_WEIGHTS[me], THREAD_WEIGHTS[G_W + line]);
#else
  F2 weights = fancyMul(CONST_THREAD_WEIGHTS[me], THREAD_WEIGHTS[G_W + line]);            // On nVidia, don't pollute the constant cache with line weights
#endif

  P(CFcarry) carryShuttlePtr = (P(CFcarry)) carryShuttle;
  CFcarry carry[NW+1];

#if AMDGPU
#define CarryShuttleAccess(me,i)        ((me) * NW + (i))                       // Generates denser global_load_dwordx4 instructions
//#define CarryShuttleAccess(me,i)      ((me) * 4 + (i)%4 + (i)/4 * 4*G_W)      // Also generates global_load_dwordx4 instructions and unit stride when NW=8
#else
#define CarryShuttleAccess(me,i)        ((me) + (i) * G_W)                      // nVidia likes this unit stride better
#endif

  float roundMax = 0;
  float carryMax = 0;

  // Calculate the most significant 32-bits of FRAC_BPW * the word index.  Also add FRAC_BPW_HI to test first biglit flag.
  u32 word_index = (me * H + line) * 2;
  u32 frac_bits = word_index * FRAC_BPW_HI + mad_hi (word_index, FRAC_BPW_LO, FRAC_BPW_HI);
  const u32 frac_bits_bigstep = ((G_W * H * 2) * FRAC_BPW_HI + (u32)(((u64)(G_W * H * 2) * FRAC_BPW_LO) >> 32));

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  F invBase = optionalDouble(weights.x);
  
  for (u32 i = 0; i < NW; ++i) {
    F invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)));
    F invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP));

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits + i * frac_bits_bigstep <= FRAC_BPW_HI;
    bool biglit1 = frac_bits + i * frac_bits_bigstep >= -FRAC_BPW_HI;   // Same as frac_bits + i * frac_bits_bigstep + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(u[i]), U2(invWeight1, invWeight2),
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);
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

  // Tell next line that its carries are ready
  if (gr < H) {
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar();
    if (me == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (me % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + me / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Line zero will be redone when gr == H
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Calculate inverse weights
  F base = optionalHalve(weights.y);
  for (u32 i = 0; i < NW; ++i) {
    F weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    F weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP));
    u[i] = U2(weight1, weight2);
  }

  // Wait until our carries are ready
#if OLD_FENCE
  if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
  // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
  bar();
  read_mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me == 0) ready[gr - 1] = 0;
#else
  u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
  if (me % WAVEFRONT == 0) {
    do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
  __asm("s_setprio 1");
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
    bool biglit0 = frac_bits + i * frac_bits_bigstep <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    u[i] = U2(u[i].x * wu[i].x, u[i].y * wu[i].y);
  }

  bar();

//  fft_WIDTH(lds, u, smallTrig);
  new_fft_WIDTH2(lds, u, smallTrig);

  writeCarryFusedLine(u, out, line);
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT31

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W) carryFused(P(GF31) out, CP(GF31) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, TrigGF31 smallTrig, P(uint) bufROE) {

#if 0   // fft_WIDTH uses shufl_int instead of shufl
  local GF31 lds[WIDTH / 4];
#else
  local GF31 lds[WIDTH / 2];
#endif

  GF31 u[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
  u32 line = gr % H;

#if HAS_ASM
  __asm("s_setprio 3");
#endif

  readCarryFusedLine(in, u, line);

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
#if ZEROHACK_W
  u32 zerohack = get_group_id(0) / 131072;
  new_fft_WIDTH1(lds + zerohack, u, smallTrig + zerohack);
#else
  new_fft_WIDTH1(lds, u, smallTrig);
#endif

  Word2 wu[NW];

  P(CFcarry) carryShuttlePtr = (P(CFcarry)) carryShuttle;
  CFcarry carry[NW+1];

#if AMDGPU
#define CarryShuttleAccess(me,i)        ((me) * NW + (i))                       // Generates denser global_load_dwordx4 instructions
//#define CarryShuttleAccess(me,i)      ((me) * 4 + (i)%4 + (i)/4 * 4*G_W)      // Also generates global_load_dwordx4 instructions and unit stride when NW=8
#else
#define CarryShuttleAccess(me,i)        ((me) + (i) * G_W)                      // nVidia likes this unit stride better
#endif

  u32 roundMax = 0;
  float carryMax = 0;

  u32 word_index = (me * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Weights can be applied with shifts because 2 is the 60th root GF31.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 31.
  const u32 log2_root_two = (u32) (((1ULL << 30) / NWORDS) % 31);
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits       combo.a[0]
#define weight_shift    combo.a[1]
#define combo_counter   combo.b

  const u64 combo_step = ((u64) bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 combo_bigstep = ((G_W * H * 2 - 1) * combo_step + (((u64) (G_W * H * 2 - 1) * FRAC_BPW_LO) >> 32)) % (31ULL << 32);
  combo_counter = word_index * combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  weight_shift = weight_shift % 31;
  u64 starting_combo_counter = combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = weight_shift + log2_NWORDS + 1;
  if (weight_shift > 31) weight_shift -= 31;

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  for (u32 i = 0; i < NW; ++i) {
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(u[i]), weight_shift0, weight_shift1,
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }
  combo_counter = starting_combo_counter;     // Restore starting counter for applying weights after carry propagation

#if ROE
  float fltRoundMax = (float) roundMax / (float) M31;      // For speed, roundoff was computed as 32-bit integer.  Convert to float.
  updateStats(bufROE, posROE, fltRoundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries. Only groups 0 to H-1 need to write carries out.
  // Group H is a duplicate of group 0 (producing the same results) so we don't care about group H writing out,
  // but it's fine either way.
  if (gr < H) { for (i32 i = 0; i < NW; ++i) { carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(me, i)] = carry[i]; } }

  // Tell next line that its carries are ready
  if (gr < H) {
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar();
    if (me == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (me % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + me / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Line zero will be redone when gr == H
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Wait until our carries are ready
#if OLD_FENCE
  if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
  // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
  bar();
  read_mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me == 0) ready[gr - 1] = 0;
#else
  u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
  if (me % WAVEFRONT == 0) {
    do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
  __asm("s_setprio 1");
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

  // Apply each 32 or 64 bit carry to the 2 words.  Apply weights.
  for (i32 i = 0; i < NW; ++i) {
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;
    // Generate big-word/little-word flag, propagate final carry
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    u[i] = U2(shl(make_Z31(wu[i].x), weight_shift0), shl(make_Z31(wu[i].y), weight_shift1));
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }

  bar();

  new_fft_WIDTH2(lds, u, smallTrig);

  writeCarryFusedLine(u, out, line);
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT61

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W) carryFused(P(GF61) out, CP(GF61) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, TrigGF61 smallTrig, P(uint) bufROE) {

#if 0   // fft_WIDTH uses shufl_int instead of shufl
  local GF61 lds[WIDTH / 4];
#else
  local GF61 lds[WIDTH / 2];
#endif

  GF61 u[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
  u32 line = gr % H;

#if HAS_ASM
  __asm("s_setprio 3");
#endif

  readCarryFusedLine(in, u, line);

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
#if ZEROHACK_W
  u32 zerohack = get_group_id(0) / 131072;
  new_fft_WIDTH1(lds + zerohack, u, smallTrig + zerohack);
#else
  new_fft_WIDTH1(lds, u, smallTrig);
#endif

  Word2 wu[NW];

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
#define CarryShuttleAccess(me,i)        ((me) + (i) * G_W)                      // nVidia likes this unit stride better
#endif

  u32 roundMax = 0;
  float carryMax = 0;

  u32 word_index = (me * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Weights can be applied with shifts because 2 is the 60th root GF61.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 61.
  const u32 log2_root_two = (u32) (((1ULL << 60) / NWORDS) % 61);
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 61;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits       combo.a[0]
#define weight_shift    combo.a[1]
#define combo_counter   combo.b

  const u64 combo_step = ((u64) bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 combo_bigstep = ((G_W * H * 2 - 1) * combo_step + (((u64) (G_W * H * 2 - 1) * FRAC_BPW_LO) >> 32)) % (61ULL << 32);
  combo_counter = word_index * combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  weight_shift = weight_shift % 61;
  u64 starting_combo_counter = combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = weight_shift + log2_NWORDS + 1;
  if (weight_shift > 61) weight_shift -= 61;

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  for (u32 i = 0; i < NW; ++i) {
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(u[i]), weight_shift0, weight_shift1,
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 61) weight_shift -= 61;
  }
  combo_counter = starting_combo_counter;     // Restore starting counter for applying weights after carry propagation

#if ROE
  float fltRoundMax = (float) roundMax / (float) (M61 >> 32);      // For speed, roundoff was computed as 32-bit integer.  Convert to float.
  updateStats(bufROE, posROE, fltRoundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries. Only groups 0 to H-1 need to write carries out.
  // Group H is a duplicate of group 0 (producing the same results) so we don't care about group H writing out,
  // but it's fine either way.
  if (gr < H) { for (i32 i = 0; i < NW; ++i) { carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(me, i)] = carry[i]; } }

  // Tell next line that its carries are ready
  if (gr < H) {
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar();
    if (me == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (me % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + me / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Line zero will be redone when gr == H
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Wait until our carries are ready
#if OLD_FENCE
  if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
  // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
  bar();
  read_mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me == 0) ready[gr - 1] = 0;
#else
  u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
  if (me % WAVEFRONT == 0) {
    do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
  __asm("s_setprio 1");
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

  // Apply each 32 or 64 bit carry to the 2 words.  Apply weights.
  for (i32 i = 0; i < NW; ++i) {
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;
    // Generate big-word/little-word flag, propagate final carry
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    u[i] = U2(shl(make_Z61(wu[i].x), weight_shift0), shl(make_Z61(wu[i].y), weight_shift1));
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 61) weight_shift -= 61;
  }

  bar();

  new_fft_WIDTH2(lds, u, smallTrig);

  writeCarryFusedLine(u, out, line);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP64 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT6431

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W) carryFused(P(T2) out, CP(T2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
                       CP(u32) bits, ConstBigTab CONST_THREAD_WEIGHTS, BigTab THREAD_WEIGHTS, P(uint) bufROE) {

  local T2 lds[WIDTH / 2];
  local GF31 *lds31 = (local GF31 *) lds;

  T2 u[NW];
  GF31 u31[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
  u32 line = gr % H;

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);

#if HAS_ASM
  __asm("s_setprio 3");
#endif

  readCarryFusedLine(in, u, line);
  readCarryFusedLine(in31, u31, line);

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
#if ZEROHACK_W
  u32 zerohack = get_group_id(0) / 131072;
  new_fft_WIDTH1(lds + zerohack, u, smallTrig + zerohack);
  bar();
  new_fft_WIDTH1(lds31 + zerohack, u31, smallTrig31 + zerohack);
#else
  new_fft_WIDTH1(lds, u, smallTrig);
  bar();
  new_fft_WIDTH1(lds31, u31, smallTrig31);
#endif

  Word2 wu[NW];
#if AMDGPU
  T2 weights = fancyMul(THREAD_WEIGHTS[me], THREAD_WEIGHTS[G_W + line]);
#else
  T2 weights = fancyMul(CONST_THREAD_WEIGHTS[me], THREAD_WEIGHTS[G_W + line]);            // On nVidia, don't pollute the constant cache with line weights
#endif
  P(i64) carryShuttlePtr = (P(i64)) carryShuttle;
  i64 carry[NW+1];

#if AMDGPU
#define CarryShuttleAccess(me,i)        ((me) * NW + (i))                       // Generates denser global_load_dwordx4 instructions
//#define CarryShuttleAccess(me,i)      ((me) * 4 + (i)%4 + (i)/4 * 4*G_W)      // Also generates global_load_dwordx4 instructions and unit stride when NW=8
#else
#define CarryShuttleAccess(me,i)        ((me) + (i) * G_W)                      // nVidia likes this unit stride better
#endif

  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (me * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 31.
  const u32 log2_root_two = (u32) (((1ULL << 30) / NWORDS) % 31);
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits       combo.a[0]
#define weight_shift    combo.a[1]
#define combo_counter   combo.b

  const u64 combo_step = ((u64) bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 combo_bigstep = ((G_W * H * 2 - 1) * combo_step + (((u64) (G_W * H * 2 - 1) * FRAC_BPW_LO) >> 32)) % (31ULL << 32);
  combo_counter = word_index * combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  weight_shift = weight_shift % 31;
  u64 starting_combo_counter = combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = weight_shift + log2_NWORDS + 1;
  if (weight_shift > 31) weight_shift -= 31;

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  T invBase = optionalDouble(weights.x);
  for (u32 i = 0; i < NW; ++i) {
    // Generate the FP64 weights and second GF31 weight shift
    T invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)));
    T invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP));
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(u[i]), SWAP_XY(u31[i]), invWeight1, invWeight2, weight_shift0, weight_shift1,
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      LL != 0, (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }
  combo_counter = starting_combo_counter;     // Restore starting counter for applying weights after carry propagation

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries. Only groups 0 to H-1 need to write carries out.
  // Group H is a duplicate of group 0 (producing the same results) so we don't care about group H writing out,
  // but it's fine either way.
  if (gr < H) { for (i32 i = 0; i < NW; ++i) { carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(me, i)] = carry[i]; } }

  // Tell next line that its carries are ready
  if (gr < H) {
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar();
    if (me == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (me % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + me / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Line zero will be redone when gr == H
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Calculate inverse weights
  T base = optionalHalve(weights.y);
  for (u32 i = 0; i < NW; ++i) {
    T weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    T weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP));
    u[i] = U2(weight1, weight2);
  }

  // Wait until our carries are ready
#if OLD_FENCE
  if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
  // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
  bar();
  read_mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me == 0) ready[gr - 1] = 0;
#else
  u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
  if (me % WAVEFRONT == 0) {
    do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
  __asm("s_setprio 1");
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

  // Apply each 32 or 64 bit carry to the 2 words.  Apply weights.
  for (i32 i = 0; i < NW; ++i) {
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;
    // Generate big-word/little-word flag, propagate final carry
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    u[i] = U2(u[i].x * wu[i].x, u[i].y * wu[i].y);
    u31[i] = U2(shl(make_Z31(wu[i].x), weight_shift0), shl(make_Z31(wu[i].y), weight_shift1));

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }

  bar();

  new_fft_WIDTH2(lds, u, smallTrig);
  writeCarryFusedLine(u, out, line);

  bar();

  new_fft_WIDTH2(lds31, u31, smallTrig31);
  writeCarryFusedLine(u31, out31, line);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3231

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W) carryFused(P(T2) out, CP(T2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
                       CP(u32) bits, ConstBigTabFP32 CONST_THREAD_WEIGHTS, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {

  local F2 ldsF2[WIDTH / 2];
  local GF31 *lds31 = (local GF31 *) ldsF2;

  F2 uF2[NW];
  GF31 u31[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
  u32 line = gr % H;

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;
  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);

#if HAS_ASM
  __asm("s_setprio 3");
#endif

  readCarryFusedLine(inF2, uF2, line);
  readCarryFusedLine(in31, u31, line);

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
#if ZEROHACK_W
  u32 zerohack = get_group_id(0) / 131072;
  new_fft_WIDTH1(ldsF2 + zerohack, uF2, smallTrigF2 + zerohack);
  bar();
  new_fft_WIDTH1(lds31 + zerohack, u31, smallTrig31 + zerohack);
#else
  new_fft_WIDTH1(ldsF2, uF2, smallTrigF2);
  bar();
  new_fft_WIDTH1(lds31, u31, smallTrig31);
#endif

  Word2 wu[NW];
#if AMDGPU
  F2 weights = fancyMul(THREAD_WEIGHTS[me], THREAD_WEIGHTS[G_W + line]);
#else
  F2 weights = fancyMul(CONST_THREAD_WEIGHTS[me], THREAD_WEIGHTS[G_W + line]);            // On nVidia, don't pollute the constant cache with line weights
#endif
  P(i32) carryShuttlePtr = (P(i32)) carryShuttle;
  i32 carry[NW+1];

#if AMDGPU
#define CarryShuttleAccess(me,i)        ((me) * NW + (i))                       // Generates denser global_load_dwordx4 instructions
//#define CarryShuttleAccess(me,i)      ((me) * 4 + (i)%4 + (i)/4 * 4*G_W)      // Also generates global_load_dwordx4 instructions and unit stride when NW=8
#else
#define CarryShuttleAccess(me,i)        ((me) + (i) * G_W)                      // nVidia likes this unit stride better
#endif

  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (me * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 31.
  const u32 log2_root_two = (u32) (((1ULL << 30) / NWORDS) % 31);
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits       combo.a[0]
#define weight_shift    combo.a[1]
#define combo_counter   combo.b

  const u64 combo_step = ((u64) bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 combo_bigstep = ((G_W * H * 2 - 1) * combo_step + (((u64) (G_W * H * 2 - 1) * FRAC_BPW_LO) >> 32)) % (31ULL << 32);
  combo_counter = word_index * combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  weight_shift = weight_shift % 31;
  u64 starting_combo_counter = combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = weight_shift + log2_NWORDS + 1;
  if (weight_shift > 31) weight_shift -= 31;

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  F invBase = optionalDouble(weights.x);
  for (u32 i = 0; i < NW; ++i) {
    // Generate the FP32 weights and second GF31 weight shift
    F invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)));
    F invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP));
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(uF2[i]), SWAP_XY(u31[i]), invWeight1, invWeight2, weight_shift0, weight_shift1,
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }
  combo_counter = starting_combo_counter;     // Restore starting counter for applying weights after carry propagation

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries. Only groups 0 to H-1 need to write carries out.
  // Group H is a duplicate of group 0 (producing the same results) so we don't care about group H writing out,
  // but it's fine either way.
  if (gr < H) { for (i32 i = 0; i < NW; ++i) { carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(me, i)] = carry[i]; } }

  // Tell next line that its carries are ready
  if (gr < H) {
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar();
    if (me == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (me % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + me / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Line zero will be redone when gr == H
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Calculate inverse weights
  F base = optionalHalve(weights.y);
  for (u32 i = 0; i < NW; ++i) {
    F weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    F weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP));
    uF2[i] = U2(weight1, weight2);
  }

  // Wait until our carries are ready
#if OLD_FENCE
  if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
  // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
  bar();
  read_mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me == 0) ready[gr - 1] = 0;
#else
  u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
  if (me % WAVEFRONT == 0) {
    do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
  __asm("s_setprio 1");
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

  // Apply each 32 or 64 bit carry to the 2 words.  Apply weights.
  for (i32 i = 0; i < NW; ++i) {
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;
    // Generate big-word/little-word flag, propagate final carry
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    uF2[i] = U2(uF2[i].x * wu[i].x, uF2[i].y * wu[i].y);
    u31[i] = U2(shl(make_Z31(wu[i].x), weight_shift0), shl(make_Z31(wu[i].y), weight_shift1));

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }

  bar();

  new_fft_WIDTH2(ldsF2, uF2, smallTrigF2);
  writeCarryFusedLine(uF2, outF2, line);

  bar();

  new_fft_WIDTH2(lds31, u31, smallTrig31);
  writeCarryFusedLine(u31, out31, line);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M61^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3261

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W) carryFused(P(T2) out, CP(T2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
                       CP(u32) bits, ConstBigTabFP32 CONST_THREAD_WEIGHTS, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {

  local GF61 lds61[WIDTH / 2];
  local F2 *ldsF2 = (local F2 *) lds61;

  F2 uF2[NW];
  GF61 u61[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
  u32 line = gr % H;

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTWTRIGGF61);

#if HAS_ASM
  __asm("s_setprio 3");
#endif

  readCarryFusedLine(inF2, uF2, line);
  readCarryFusedLine(in61, u61, line);

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
#if ZEROHACK_W
  u32 zerohack = get_group_id(0) / 131072;
  new_fft_WIDTH1(ldsF2 + zerohack, uF2, smallTrigF2 + zerohack);
  bar();
  new_fft_WIDTH1(lds61 + zerohack, u61, smallTrig61 + zerohack);
#else
  new_fft_WIDTH1(ldsF2, uF2, smallTrigF2);
  bar();
  new_fft_WIDTH1(lds61, u61, smallTrig61);
#endif

  Word2 wu[NW];
#if AMDGPU
  F2 weights = fancyMul(THREAD_WEIGHTS[me], THREAD_WEIGHTS[G_W + line]);
#else
  F2 weights = fancyMul(CONST_THREAD_WEIGHTS[me], THREAD_WEIGHTS[G_W + line]);            // On nVidia, don't pollute the constant cache with line weights
#endif
  P(i64) carryShuttlePtr = (P(i64)) carryShuttle;
  i64 carry[NW+1];

#if AMDGPU
#define CarryShuttleAccess(me,i)        ((me) * NW + (i))                       // Generates denser global_load_dwordx4 instructions
//#define CarryShuttleAccess(me,i)      ((me) * 4 + (i)%4 + (i)/4 * 4*G_W)      // Also generates global_load_dwordx4 instructions and unit stride when NW=8
#else
#define CarryShuttleAccess(me,i)        ((me) + (i) * G_W)                      // nVidia likes this unit stride better
#endif

  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (me * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 61.
  const u32 log2_root_two = (u32) (((1ULL << 60) / NWORDS) % 61);
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 61;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits       combo.a[0]
#define weight_shift    combo.a[1]
#define combo_counter   combo.b

  const u64 combo_step = ((u64) bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 combo_bigstep = ((G_W * H * 2 - 1) * combo_step + (((u64) (G_W * H * 2 - 1) * FRAC_BPW_LO) >> 32)) % (61ULL << 32);
  combo_counter = word_index * combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  weight_shift = weight_shift % 61;
  u64 starting_combo_counter = combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = weight_shift + log2_NWORDS + 1;
  if (weight_shift > 61) weight_shift -= 61;

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  F invBase = optionalDouble(weights.x);
  for (u32 i = 0; i < NW; ++i) {
    // Generate the FP32 weights and second GF61 weight shift
    F invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)));
    F invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP));
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(uF2[i]), SWAP_XY(u61[i]), invWeight1, invWeight2, weight_shift0, weight_shift1,
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      LL != 0, (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 61) weight_shift -= 61;
  }
  combo_counter = starting_combo_counter;     // Restore starting counter for applying weights after carry propagation

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries. Only groups 0 to H-1 need to write carries out.
  // Group H is a duplicate of group 0 (producing the same results) so we don't care about group H writing out,
  // but it's fine either way.
  if (gr < H) { for (i32 i = 0; i < NW; ++i) { carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(me, i)] = carry[i]; } }

  // Tell next line that its carries are ready
  if (gr < H) {
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar();
    if (me == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (me % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + me / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Line zero will be redone when gr == H
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Calculate inverse weights
  F base = optionalHalve(weights.y);
  for (u32 i = 0; i < NW; ++i) {
    F weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    F weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP));
    uF2[i] = U2(weight1, weight2);
  }

  // Wait until our carries are ready
#if OLD_FENCE
  if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
  // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
  bar();
  read_mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me == 0) ready[gr - 1] = 0;
#else
  u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
  if (me % WAVEFRONT == 0) {
    do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
  __asm("s_setprio 1");
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

  // Apply each 32 or 64 bit carry to the 2 words.  Apply weights.
  for (i32 i = 0; i < NW; ++i) {
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;
    // Generate big-word/little-word flag, propagate final carry
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    uF2[i] = U2(uF2[i].x * wu[i].x, uF2[i].y * wu[i].y);
    u61[i] = U2(shl(make_Z61(wu[i].x), weight_shift0), shl(make_Z61(wu[i].y), weight_shift1));

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 61) weight_shift -= 61;
  }

  bar();

  new_fft_WIDTH2(ldsF2, uF2, smallTrigF2);
  writeCarryFusedLine(uF2, outF2, line);

  bar();

  new_fft_WIDTH2(lds61, u61, smallTrig61);
  writeCarryFusedLine(u61, out61, line);
}


/**************************************************************************/
/*    Similar to above, but for an NTT based on GF(M31^2)*GF(M61^2)       */
/**************************************************************************/

#elif FFT_TYPE == FFT3161

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W) carryFused(P(T2) out, CP(T2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, Trig smallTrig, P(uint) bufROE) {

#if 0   // fft_WIDTH uses shufl_int instead of shufl
  local GF61 lds61[WIDTH / 4];
#else
  local GF61 lds61[WIDTH / 2];
#endif
  local GF31 *lds31 = (local GF31 *) lds61;

  GF31 u31[NW];
  GF61 u61[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
  u32 line = gr % H;

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTWTRIGGF61);

#if HAS_ASM
  __asm("s_setprio 3");
#endif

  readCarryFusedLine(in31, u31, line);
  readCarryFusedLine(in61, u61, line);

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
#if ZEROHACK_W
  u32 zerohack = get_group_id(0) / 131072;
  new_fft_WIDTH1(lds31 + zerohack, u31, smallTrig31 + zerohack);
  bar();
  new_fft_WIDTH1(lds61 + zerohack, u61, smallTrig61 + zerohack);
#else
  new_fft_WIDTH1(lds31, u31, smallTrig31);
  bar();
  new_fft_WIDTH1(lds61, u61, smallTrig61);
#endif

  Word2 wu[NW];
  P(i64) carryShuttlePtr = (P(i64)) carryShuttle;
  i64 carry[NW+1];

#if AMDGPU
#define CarryShuttleAccess(me,i)        ((me) * NW + (i))                       // Generates denser global_load_dwordx4 instructions
//#define CarryShuttleAccess(me,i)      ((me) * 4 + (i)%4 + (i)/4 * 4*G_W)      // Also generates global_load_dwordx4 instructions and unit stride when NW=8
#else
#define CarryShuttleAccess(me,i)        ((me) + (i) * G_W)                      // nVidia likes this unit stride better
#endif

  u32 roundMax = 0;
  float carryMax = 0;

  u32 word_index = (me * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 31.
  const u32 m31_log2_root_two = (u32) (((1ULL << 30) / NWORDS) % 31);
  const u32 m31_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m31_log2_root_two % 31;
  const u32 m31_bigword_weight_shift_minus1 = (m31_bigword_weight_shift + 30) % 31;
  const u32 m61_log2_root_two = (u32) (((1ULL << 60) / NWORDS) % 61);
  const u32 m61_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m61_log2_root_two % 61;
  const u32 m61_bigword_weight_shift_minus1 = (m61_bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } m31_combo, m61_combo;
#define frac_bits           m31_combo.a[0]
#define m31_weight_shift    m31_combo.a[1]
#define m31_combo_counter   m31_combo.b
#define m61_weight_shift    m61_combo.a[1]
#define m61_combo_counter   m61_combo.b

  const u64 m31_combo_step = ((u64) m31_bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 m31_combo_bigstep = ((G_W * H * 2 - 1) * m31_combo_step + (((u64) (G_W * H * 2 - 1) * FRAC_BPW_LO) >> 32)) % (31ULL << 32);
  m31_combo_counter = word_index * m31_combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  m31_weight_shift = m31_weight_shift % 31;
  u64 m31_starting_combo_counter = m31_combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation
  const u64 m61_combo_step = ((u64) m61_bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 m61_combo_bigstep = ((G_W * H * 2 - 1) * m61_combo_step + (((u64) (G_W * H * 2 - 1) * FRAC_BPW_LO) >> 32)) % (61ULL << 32);
  m61_combo_counter = word_index * m61_combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  m61_weight_shift = m61_weight_shift % 61;
  u64 m61_starting_combo_counter = m61_combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift + log2_NWORDS + 1);
  m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift + log2_NWORDS + 1);

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  for (u32 i = 0; i < NW; ++i) {
    // Generate the second weight shifts
    u32 m31_weight_shift0 = m31_weight_shift;
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    u32 m31_weight_shift1 = m31_weight_shift;
    u32 m61_weight_shift0 = m61_weight_shift;
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
    u32 m61_weight_shift1 = m61_weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(u31[i]), SWAP_XY(u61[i]), m31_weight_shift0, m31_weight_shift1, m61_weight_shift0, m61_weight_shift1,
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      LL != 0, (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_bigstep;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_bigstep;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }
  m31_combo_counter = m31_starting_combo_counter;     // Restore starting counter for applying weights after carry propagation
  m61_combo_counter = m61_starting_combo_counter;

#if ROE
  float fltRoundMax = (float) roundMax / (float) 0x1FFFFFFF;      // For speed, roundoff was computed as 32-bit integer.  Convert to float - divide by M61.
  updateStats(bufROE, posROE, fltRoundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries. Only groups 0 to H-1 need to write carries out.
  // Group H is a duplicate of group 0 (producing the same results) so we don't care about group H writing out,
  // but it's fine either way.
  if (gr < H) { for (i32 i = 0; i < NW; ++i) { carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(me, i)] = carry[i]; } }

  // Tell next line that its carries are ready
  if (gr < H) {
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar();
    if (me == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (me % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + me / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Line zero will be redone when gr == H
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Wait until our carries are ready
#if OLD_FENCE
  if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
  // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
  bar();
  read_mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me == 0) ready[gr - 1] = 0;
#else
  u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
  if (me % WAVEFRONT == 0) {
    do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
  __asm("s_setprio 1");
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

  // Apply each 32 or 64 bit carry to the 2 words.  Apply weights.
  for (i32 i = 0; i < NW; ++i) {
    // Generate the second weight shifts
    u32 m31_weight_shift0 = m31_weight_shift;
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    u32 m31_weight_shift1 = m31_weight_shift;
    u32 m61_weight_shift0 = m61_weight_shift;
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
    u32 m61_weight_shift1 = m61_weight_shift;
    // Generate big-word/little-word flag, propagate final carry
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    u31[i] = U2(shl(make_Z31(wu[i].x), m31_weight_shift0), shl(make_Z31(wu[i].y), m31_weight_shift1));
    u61[i] = U2(shl(make_Z61(wu[i].x), m61_weight_shift0), shl(make_Z61(wu[i].y), m61_weight_shift1));

    // Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_bigstep;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_bigstep;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }

  bar();

  new_fft_WIDTH2(lds31, u31, smallTrig31);
  writeCarryFusedLine(u31, out31, line);

  bar();

  new_fft_WIDTH2(lds61, u61, smallTrig61);
  writeCarryFusedLine(u61, out61, line);
}


/******************************************************************************/
/*  Similar to above, but for a hybrid FFT based on FP32*GF(M31^2)*GF(M61^2)  */
/******************************************************************************/

#elif FFT_TYPE == FFT323161

// The "carryFused" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway forwarding" (forwarding carry data from one workgroup to the next)
KERNEL(G_W) carryFused(P(T2) out, CP(T2) in, u32 posROE, P(i64) carryShuttle, P(u32) ready, Trig smallTrig,
                       CP(u32) bits, ConstBigTabFP32 CONST_THREAD_WEIGHTS, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {

#if 0   // fft_WIDTH uses shufl_int instead of shufl
  local GF61 lds61[WIDTH / 4];
#else
  local GF61 lds61[WIDTH / 2];
#endif
  local F2 *ldsF2 = (local F2 *) lds61;
  local GF31 *lds31 = (local GF31 *) lds61;

  F2 uF2[NW];
  GF31 u31[NW];
  GF61 u61[NW];

  u32 gr = get_group_id(0);
  u32 me = get_local_id(0);

  u32 H = BIG_HEIGHT;
  u32 line = gr % H;

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;
  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTWTRIGGF61);

#if HAS_ASM
  __asm("s_setprio 3");
#endif

  readCarryFusedLine(inF2, uF2, line);
  readCarryFusedLine(in31, u31, line);
  readCarryFusedLine(in61, u61, line);

// Try this weird FFT_width call that adds a "hidden zero" when unrolling.  This prevents the compiler from finding
// common sub-expressions to re-use in the second fft_WIDTH call.  Re-using this data requires dozens of VGPRs
// which causes a terrible reduction in occupancy.
#if ZEROHACK_W
  u32 zerohack = get_group_id(0) / 131072;
  new_fft_WIDTH1(ldsF2 + zerohack, uF2, smallTrigF2 + zerohack);
  bar();
  new_fft_WIDTH1(lds31 + zerohack, u31, smallTrig31 + zerohack);
  bar();
  new_fft_WIDTH1(lds61 + zerohack, u61, smallTrig61 + zerohack);
#else
  new_fft_WIDTH1(ldsF2, uF2, smallTrigF2);
  bar();
  new_fft_WIDTH1(lds31, u31, smallTrig31);
  bar();
  new_fft_WIDTH1(lds61, u61, smallTrig61);
#endif

  Word2 wu[NW];
#if AMDGPU
  F2 weights = fancyMul(THREAD_WEIGHTS[me], THREAD_WEIGHTS[G_W + line]);
#else
  F2 weights = fancyMul(CONST_THREAD_WEIGHTS[me], THREAD_WEIGHTS[G_W + line]);            // On nVidia, don't pollute the constant cache with line weights
#endif
  P(i64) carryShuttlePtr = (P(i64)) carryShuttle;
  i64 carry[NW+1];

#if AMDGPU
#define CarryShuttleAccess(me,i)        ((me) * NW + (i))                       // Generates denser global_load_dwordx4 instructions
//#define CarryShuttleAccess(me,i)      ((me) * 4 + (i)%4 + (i)/4 * 4*G_W)      // Also generates global_load_dwordx4 instructions and unit stride when NW=8
#else
#define CarryShuttleAccess(me,i)        ((me) + (i) * G_W)                      // nVidia likes this unit stride better
#endif

  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (me * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 31.
  const u32 m31_log2_root_two = (u32) (((1ULL << 30) / NWORDS) % 31);
  const u32 m31_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m31_log2_root_two % 31;
  const u32 m31_bigword_weight_shift_minus1 = (m31_bigword_weight_shift + 30) % 31;
  const u32 m61_log2_root_two = (u32) (((1ULL << 60) / NWORDS) % 61);
  const u32 m61_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m61_log2_root_two % 61;
  const u32 m61_bigword_weight_shift_minus1 = (m61_bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } m31_combo, m61_combo;
#define frac_bits           m31_combo.a[0]
#define m31_weight_shift    m31_combo.a[1]
#define m31_combo_counter   m31_combo.b
#define m61_weight_shift    m61_combo.a[1]
#define m61_combo_counter   m61_combo.b

  const u64 m31_combo_step = ((u64) m31_bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 m31_combo_bigstep = ((G_W * H * 2 - 1) * m31_combo_step + (((u64) (G_W * H * 2 - 1) * FRAC_BPW_LO) >> 32)) % (31ULL << 32);
  m31_combo_counter = word_index * m31_combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  m31_weight_shift = m31_weight_shift % 31;
  u64 m31_starting_combo_counter = m31_combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation
  const u64 m61_combo_step = ((u64) m61_bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 m61_combo_bigstep = ((G_W * H * 2 - 1) * m61_combo_step + (((u64) (G_W * H * 2 - 1) * FRAC_BPW_LO) >> 32)) % (61ULL << 32);
  m61_combo_counter = word_index * m61_combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  m61_weight_shift = m61_weight_shift % 61;
  u64 m61_starting_combo_counter = m61_combo_counter;     // Save starting counter before adding log2_NWORDS+1 for applying weights after carry propagation

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift + log2_NWORDS + 1);
  m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift + log2_NWORDS + 1);

  // Apply the inverse weights and carry propagate pairs to generate the output carries

  F invBase = optionalDouble(weights.x);
  for (u32 i = 0; i < NW; ++i) {
    // Generate the FP32 weights and second GF31 and GF61 weight shift
    F invWeight1 = i == 0 ? invBase : optionalDouble(fancyMul(invBase, iweightStep(i)));
    F invWeight2 = optionalDouble(fancyMul(invWeight1, IWEIGHT_STEP));
    u32 m31_weight_shift0 = m31_weight_shift;
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    u32 m31_weight_shift1 = m31_weight_shift;
    u32 m61_weight_shift0 = m61_weight_shift;
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
    u32 m61_weight_shift1 = m61_weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Apply the inverse weights, optionally compute roundoff error, and convert to integer.  Also apply MUL3 here.
    // Then propagate carries through two words (the first carry does not have to be accurately calculated because it will
    // be accurately calculated by carryFinal later on).  The second carry must be accurate for output to the carry shuttle.
    wu[i] = weightAndCarryPairSloppy(SWAP_XY(uF2[i]), SWAP_XY(u31[i]), SWAP_XY(u61[i]), invWeight1, invWeight2, m31_weight_shift0, m31_weight_shift1, m61_weight_shift0, m61_weight_shift1,
                      // For an LL test, add -2 as the very initial "carry in"
                      // We'd normally use logical &&, but the compiler whines with warning and bitwise fixes it
                      LL != 0, (LL & (i == 0) & (line==0) & (me == 0)) ? -2 : 0, biglit0, biglit1, &carry[i], &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_bigstep;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_bigstep;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }
  m31_combo_counter = m31_starting_combo_counter;     // Restore starting counter for applying weights after carry propagation
  m61_combo_counter = m61_starting_combo_counter;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif STATS & (1 << MUL3)
  updateStats(bufROE, posROE, carryMax);
#endif

  // Write out our carries. Only groups 0 to H-1 need to write carries out.
  // Group H is a duplicate of group 0 (producing the same results) so we don't care about group H writing out,
  // but it's fine either way.
  if (gr < H) { for (i32 i = 0; i < NW; ++i) { carryShuttlePtr[gr * WIDTH + CarryShuttleAccess(me, i)] = carry[i]; } }

  // Tell next line that its carries are ready
  if (gr < H) {
#if OLD_FENCE
    // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    bar();
    if (me == 0) { atomic_store((atomic_uint *) &ready[gr], 1); }
#else
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    if (me % WAVEFRONT == 0) { 
      u32 pos = gr * (G_W / WAVEFRONT) + me / WAVEFRONT;
      atomic_store((atomic_uint *) &ready[pos], 1);
    }
#endif
  }

  // Line zero will be redone when gr == H
  if (gr == 0) { return; }

  // Do some work while our carries may not be ready
#if HAS_ASM
  __asm("s_setprio 0");
#endif

  // Calculate inverse weights
  F base = optionalHalve(weights.y);
  for (u32 i = 0; i < NW; ++i) {
    F weight1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    F weight2 = optionalHalve(fancyMul(weight1, WEIGHT_STEP));
    uF2[i] = U2(weight1, weight2);
  }

  // Wait until our carries are ready
#if OLD_FENCE
  if (me == 0) { do { spin(); } while(!atomic_load_explicit((atomic_uint *) &ready[gr - 1], memory_order_relaxed, memory_scope_device)); }
  // work_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
  bar();
  read_mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me == 0) ready[gr - 1] = 0;
#else
  u32 pos = (gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT;
  if (me % WAVEFRONT == 0) {
    do { spin(); } while(atomic_load_explicit((atomic_uint *) &ready[pos], memory_order_relaxed, memory_scope_device) == 0);
  }
  mem_fence(CLK_GLOBAL_MEM_FENCE);
  // Clear carry ready flag for next iteration
  if (me % WAVEFRONT == 0) ready[(gr - 1) * (G_W / WAVEFRONT) + me / WAVEFRONT] = 0;
#endif
#if HAS_ASM
  __asm("s_setprio 1");
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

  // Apply each 32 or 64 bit carry to the 2 words.  Apply weights.
  for (i32 i = 0; i < NW; ++i) {
    // Generate the second weight shifts
    u32 m31_weight_shift0 = m31_weight_shift;
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    u32 m31_weight_shift1 = m31_weight_shift;
    u32 m61_weight_shift0 = m61_weight_shift;
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
    u32 m61_weight_shift1 = m61_weight_shift;
    // Generate big-word/little-word flag, propagate final carry
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    wu[i] = carryFinal(wu[i], carry[i], biglit0);
    uF2[i] = U2(uF2[i].x * wu[i].x, uF2[i].y * wu[i].y);
    u31[i] = U2(shl(make_Z31(wu[i].x), m31_weight_shift0), shl(make_Z31(wu[i].y), m31_weight_shift1));
    u61[i] = U2(shl(make_Z61(wu[i].x), m61_weight_shift0), shl(make_Z61(wu[i].y), m61_weight_shift1));

    // Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_bigstep;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_bigstep;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }

  bar();

  new_fft_WIDTH2(ldsF2, uF2, smallTrigF2);
  writeCarryFusedLine(uF2, outF2, line);

  bar();

  new_fft_WIDTH2(lds31, u31, smallTrig31);
  writeCarryFusedLine(u31, out31, line);

  bar();

  new_fft_WIDTH2(lds61, u61, smallTrig61);
  writeCarryFusedLine(u61, out61, line);
}


#else
error - missing CarryFused kernel implementation
#endif

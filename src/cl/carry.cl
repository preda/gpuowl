// Copyright (C) Mihai Preda

#include "carryutil.cl"
#include "weight.cl"

#if FFT_TYPE == FFT64

// Carry propagation with optional MUL-3, over CARRY_LEN words.
// Input arrives with real and imaginary values swapped and weighted.

KERNEL(G_W) carry(P(Word2) out, CP(T2) in, u32 posROE, P(CarryABM) carryOut, BigTab THREAD_WEIGHTS, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;

  // & vs. && to workaround spurious warning
  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  // Calculate the most significant 32-bits of FRAC_BPW * the index of the FFT word.  Also add FRAC_BPW_HI to test first biglit flag.
  u32 line = gy * CARRY_LEN;
  u32 fft_word_index = (gx * G_W * H + me * H + line) * 2;
  u32 frac_bits = fft_word_index * FRAC_BPW_HI + mad_hi (fft_word_index, FRAC_BPW_LO, FRAC_BPW_HI);

  T base = optionalDouble(fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx)));

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;
    T w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + gy * CARRY_LEN + i].x));
    T w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP));
    bool biglit0 = frac_bits + (2*i) * FRAC_BPW_HI <= FRAC_BPW_HI;
    bool biglit1 = frac_bits + (2*i) * FRAC_BPW_HI >= -FRAC_BPW_HI;   // Same as frac_bits + (2*i) * FRAC_BPW_HI + FRAC_BPW_HI <= FRAC_BPW_HI;
    out[p] = weightAndCarryPair(SWAP_XY(in[p]), U2(w1, w2), carry, biglit0, biglit1, &carry, &roundMax, &carryMax);
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#elif FFT_TYPE == FFT32

// Carry propagation with optional MUL-3, over CARRY_LEN words.
// Input arrives with real and imaginary values swapped and weighted.

KERNEL(G_W) carry(P(Word2) out, CP(F2) in, u32 posROE, P(CarryABM) carryOut, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;

  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  // Calculate the most significant 32-bits of FRAC_BPW * the index of the FFT word.  Also add FRAC_BPW_HI to test first biglit flag.
  u32 line = gy * CARRY_LEN;
  u32 fft_word_index = (gx * G_W * H + me * H + line) * 2;
  u32 frac_bits = fft_word_index * FRAC_BPW_HI + mad_hi (fft_word_index, FRAC_BPW_LO, FRAC_BPW_HI);

  F base = optionalDouble(fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx)));

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;
    F w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + gy * CARRY_LEN + i].x));
    F w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP));
    bool biglit0 = frac_bits + (2*i) * FRAC_BPW_HI <= FRAC_BPW_HI;
    bool biglit1 = frac_bits + (2*i) * FRAC_BPW_HI >= -FRAC_BPW_HI;   // Same as frac_bits + (2*i) * FRAC_BPW_HI + FRAC_BPW_HI <= FRAC_BPW_HI;
    out[p] = weightAndCarryPair(SWAP_XY(in[p]), U2(w1, w2), carry, biglit0, biglit1, &carry, &roundMax, &carryMax);
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT31

KERNEL(G_W) carry(P(Word2) out, CP(GF31) in, u32 posROE, P(CarryABM) carryOut, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;
  u32 line = gy * CARRY_LEN;

  // & vs. && to workaround spurious warning
  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  u32 roundMax = 0;
  float carryMax = 0;

  u32 word_index = (gx * G_W * H + me * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Weights can be applied with shifts because 2 is the 30th root GF31.
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
  combo_counter = word_index * combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = (weight_shift + log2_NWORDS + 1) % 31;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;

    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Compute result
    out[p] = weightAndCarryPair(SWAP_XY(in[p]), weight_shift0, weight_shift1, carry, biglit0, biglit1, &carry, &roundMax, &carryMax);
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  float fltRoundMax = (float) roundMax / (float) M31;      // For speed, roundoff was computed as 32-bit integer.  Convert to float.
  updateStats(bufROE, posROE, fltRoundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT61

KERNEL(G_W) carry(P(Word2) out, CP(GF61) in, u32 posROE, P(CarryABM) carryOut, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;
  u32 line = gy * CARRY_LEN;

  // & vs. && to workaround spurious warning
  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  u32 roundMax = 0;
  float carryMax = 0;

  u32 word_index = (gx * G_W * H + me * H + line) * 2;

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
  combo_counter = word_index * combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = (weight_shift + log2_NWORDS + 1) % 61;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;

    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Compute result
    out[p] = weightAndCarryPair(SWAP_XY(in[p]), weight_shift0, weight_shift1, carry, biglit0, biglit1, &carry, &roundMax, &carryMax);
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  float fltRoundMax = (float) roundMax / (float) (M61 >> 32);      // For speed, roundoff was computed as 32-bit integer.  Convert to float.
  updateStats(bufROE, posROE, fltRoundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP64 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT6431

KERNEL(G_W) carry(P(Word2) out, CP(T2) in, u32 posROE, P(CarryABM) carryOut, BigTab THREAD_WEIGHTS, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;
  u32 line = gy * CARRY_LEN;

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);

  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (gx * G_W * H + me * H + line) * 2;

  T base = optionalDouble(fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx)));

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
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
  combo_counter = word_index * combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = (weight_shift + log2_NWORDS + 1) % 31;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;

    // Generate the FP64 and second GF31 weight shift
    T w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + gy * CARRY_LEN + i].x));
    T w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP));
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Compute result
    out[p] = weightAndCarryPair(SWAP_XY(in[p]), SWAP_XY(in31[p]), w1, w2, weight_shift0, weight_shift1,
                                LL != 0 || i != 0, carry, biglit0, biglit1, &carry, &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3231

KERNEL(G_W) carry(P(Word2) out, CP(T2) in, u32 posROE, P(CarryABM) carryOut, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;
  u32 line = gy * CARRY_LEN;

  CP(F2) inF2 = (CP(F2)) in;
  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);

  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (gx * G_W * H + me * H + line) * 2;

  F base = optionalDouble(fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx)));

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
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
  combo_counter = word_index * combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = (weight_shift + log2_NWORDS + 1) % 31;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;

    // Generate the FP32 and second GF31 weight shift
    F w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + gy * CARRY_LEN + i].x));
    F w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP));
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Compute result
    out[p] = weightAndCarryPair(SWAP_XY(inF2[p]), SWAP_XY(in31[p]), w1, w2, weight_shift0, weight_shift1,
                                carry, biglit0, biglit1, &carry, &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M61^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3261

KERNEL(G_W) carry(P(Word2) out, CP(T2) in, u32 posROE, P(CarryABM) carryOut, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;
  u32 line = gy * CARRY_LEN;

  CP(F2) inF2 = (CP(F2)) in;
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);

  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (gx * G_W * H + me * H + line) * 2;

  F base = optionalDouble(fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx)));

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
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
  combo_counter = word_index * combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  weight_shift = (weight_shift + log2_NWORDS + 1) % 61;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;

    // Generate the FP32 and second GF61 weight shift
    F w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + gy * CARRY_LEN + i].x));
    F w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP));
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;

    // Generate big-word/little-word flags
    bool biglit0 = frac_bits <= FRAC_BPW_HI;
    bool biglit1 = frac_bits >= -FRAC_BPW_HI;   // Same as frac_bits + FRAC_BPW_HI <= FRAC_BPW_HI;

    // Compute result
    out[p] = weightAndCarryPair(SWAP_XY(inF2[p]), SWAP_XY(in61[p]), w1, w2, weight_shift0, weight_shift1,
                                LL != 0 || i != 0, carry, biglit0, biglit1, &carry, &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/**************************************************************************/
/*    Similar to above, but for an NTT based on GF(M31^2)*GF(M61^2)       */
/**************************************************************************/

#elif FFT_TYPE == FFT3161

KERNEL(G_W) carry(P(Word2) out, CP(T2) in, u32 posROE, P(CarryABM) carryOut, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;
  u32 line = gy * CARRY_LEN;

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);

  // & vs. && to workaround spurious warning
  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  u32 roundMax = 0;
  float carryMax = 0;

  u32 word_index = (gx * G_W * H + me * H + line) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
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
  m31_combo_counter = word_index * m31_combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  const u64 m61_combo_step = ((u64) m61_bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  m61_combo_counter = word_index * m61_combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  m31_weight_shift = (m31_weight_shift + log2_NWORDS + 1) % 31;
  m61_weight_shift = (m61_weight_shift + log2_NWORDS + 1) % 61;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;

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

    // Compute result
    out[p] = weightAndCarryPair(SWAP_XY(in31[p]), SWAP_XY(in61[p]), m31_weight_shift0, m31_weight_shift1, m61_weight_shift0, m61_weight_shift1,
                                LL != 0 || i != 0, carry, biglit0, biglit1, &carry, &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  float fltRoundMax = (float) roundMax / (float) 0x1FFFFFFF;      // For speed, roundoff was computed as 32-bit integer.  Convert to float.
  updateStats(bufROE, posROE, fltRoundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


/******************************************************************************/
/*  Similar to above, but for a hybrid FFT based on FP32*GF(M31^2)*GF(M61^2)  */
/******************************************************************************/

#elif FFT_TYPE == FFT323161

KERNEL(G_W) carry(P(Word2) out, CP(T2) in, u32 posROE, P(CarryABM) carryOut, BigTabFP32 THREAD_WEIGHTS, P(uint) bufROE) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;
  u32 line = gy * CARRY_LEN;

  CP(F2) inF2 = (CP(F2)) in;
  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);

  // & vs. && to workaround spurious warning
  CarryABM carry = (LL & (me == 0) & (g == 0)) ? -2 : 0;
  float roundMax = 0;
  float carryMax = 0;

  u32 word_index = (gx * G_W * H + me * H + line) * 2;

  F base = optionalDouble(fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx)));

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
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
  m31_combo_counter = word_index * m31_combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  const u64 m61_combo_step = ((u64) m61_bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  m61_combo_counter = word_index * m61_combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;

  // We also adjust shift amount for the fact that NTT returns results multiplied by 2*NWORDS.
  const u32 log2_NWORDS = (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
                          (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 : MIDDLE == 8 ? 3 : 4) +
                          (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 : SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  m31_weight_shift = (m31_weight_shift + log2_NWORDS + 1) % 31;
  m61_weight_shift = (m61_weight_shift + log2_NWORDS + 1) % 61;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;

    // Generate the FP32 and second GF31 and GF61 weight shift
    F w1 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + gy * CARRY_LEN + i].x));
    F w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP));
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

    // Compute result
    out[p] = weightAndCarryPair(SWAP_XY(inF2[p]), SWAP_XY(in31[p]), SWAP_XY(in61[p]), w1, w2, m31_weight_shift0, m31_weight_shift1, m61_weight_shift0, m61_weight_shift1,
                                LL != 0 || i != 0, carry, biglit0, biglit1, &carry, &roundMax, &carryMax);

    // Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }
  carryOut[G_W * g + me] = carry;

#if ROE
  updateStats(bufROE, posROE, roundMax);
#elif (STATS & (1 << (2 + MUL3)))
  updateStats(bufROE, posROE, carryMax);
#endif
}


#else
error - missing Carry kernel implementation
#endif

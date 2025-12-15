// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"
#include "weight.cl"
#include "fftwidth.cl"
#include "middle.cl"

#if FFT_TYPE == FFT64

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig, BigTab THREAD_WEIGHTS) {
  local T2 lds[WIDTH / 2];
  T2 u[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  in += g * WIDTH;

  T base = optionalHalve(fancyMul(THREAD_WEIGHTS[me].y, THREAD_WEIGHTS[G_W + g].y));

  for (u32 i = 0; i < NW; ++i) {
    T w1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    T w2 = optionalHalve(fancyMul(w1, WEIGHT_STEP));
    u32 p = G_W * i + me;
    u[i] = U2(in[p].x * w1, in[p].y * w2);
  }

  fft_WIDTH(lds, u, smallTrig);

  writeCarryFusedLine(u, out, g);
}


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#elif FFT_TYPE == FFT32

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(F2) out, CP(Word2) in, TrigFP32 smallTrig, BigTabFP32 THREAD_WEIGHTS) {
  local F2 lds[WIDTH / 2];
  F2 u[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  in += g * WIDTH;

  F base = optionalHalve(fancyMul(THREAD_WEIGHTS[me].y, THREAD_WEIGHTS[G_W + g].y));

  for (u32 i = 0; i < NW; ++i) {
    F w1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    F w2 = optionalHalve(fancyMul(w1, WEIGHT_STEP));
    u32 p = G_W * i + me;
    u[i] = U2(in[p].x * w1, in[p].y * w2);
  }

  fft_WIDTH(lds, u, smallTrig);

  writeCarryFusedLine(u, out, g);
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT31

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(GF31) out, CP(Word2) in, TrigGF31 smallTrig) {
  local GF31 lds[WIDTH / 2];
  GF31 u[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  in += g * WIDTH;

  u32 word_index = (me * BIG_HEIGHT + g) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  const u32 log2_root_two = (u32) (((1ULL << 30) / NWORDS) % 31);
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits           combo.a[0]
#define weight_shift        combo.a[1]
#define combo_counter       combo.b

  const u64 combo_step = ((u64) bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 combo_bigstep = ((G_W * BIG_HEIGHT * 2 - 1) * combo_step + (((u64) (G_W * BIG_HEIGHT * 2 - 1) * FRAC_BPW_LO) >> 32)) % (31ULL << 32);
  combo_counter = word_index * combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  weight_shift = weight_shift % 31;

  for (u32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;
    // Convert and weight inputs
    u[i] = U2(shl(make_Z31(in[p].x), weight_shift0), shl(make_Z31(in[p].y), weight_shift1));      // Form a GF31 from each pair of input words
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }

  fft_WIDTH(lds, u, smallTrig);

  writeCarryFusedLine(u, out, g);
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT61

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(GF61) out, CP(Word2) in, TrigGF61 smallTrig) {
  local GF61 lds[WIDTH / 2];
  GF61 u[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  in += g * WIDTH;

  u32 word_index = (me * BIG_HEIGHT + g) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Weights can be applied with shifts because 2 is the 60th root GF61.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 61.
  const u32 log2_root_two = (u32) (((1ULL << 60) / NWORDS) % 61);
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 61;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits	combo.a[0]
#define weight_shift	combo.a[1]
#define combo_counter	combo.b

  const u64 combo_step = ((u64) bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 combo_bigstep = ((G_W * BIG_HEIGHT * 2 - 1) * combo_step + (((u64) (G_W * BIG_HEIGHT * 2 - 1) * FRAC_BPW_LO) >> 32)) % (61ULL << 32);
  combo_counter = word_index * combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  weight_shift = weight_shift % 61;

  for (u32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // Generate the second weight shift
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;
    // Convert and weight input
    u[i] = U2(shl(make_Z61(in[p].x), weight_shift0), shl(make_Z61(in[p].y), weight_shift1));      // Form a GF61 from each pair of input words
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 61) weight_shift -= 61;
  }

  fft_WIDTH(lds, u, smallTrig);

  writeCarryFusedLine(u, out, g);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP64 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT6431

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig, BigTab THREAD_WEIGHTS) {
  local T2 lds[WIDTH / 2];
  local GF31 *lds31 = (local GF31 *) lds;
  T2 u[NW];
  GF31 u31[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);

  in += g * WIDTH;

  T base = optionalHalve(fancyMul(THREAD_WEIGHTS[me].y, THREAD_WEIGHTS[G_W + g].y));

  u32 word_index = (me * BIG_HEIGHT + g) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 31.
  const u32 log2_root_two = (u32) (((1ULL << 30) / NWORDS) % 31);
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits           combo.a[0]
#define weight_shift        combo.a[1]
#define combo_counter       combo.b

  const u64 combo_step = ((u64) bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 combo_bigstep = ((G_W * BIG_HEIGHT * 2 - 1) * combo_step + (((u64) (G_W * BIG_HEIGHT * 2 - 1) * FRAC_BPW_LO) >> 32)) % (31ULL << 32);
  combo_counter = word_index * combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  weight_shift = weight_shift % 31;

  for (u32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // Generate the FP64 weights and the second GF31 weight shift
    T w1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    T w2 = optionalHalve(fancyMul(w1, WEIGHT_STEP));
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;
    // Convert and weight input
    u[i] = U2(in[p].x * w1, in[p].y * w2);
    u31[i] = U2(shl(make_Z31(in[p].x), weight_shift0), shl(make_Z31(in[p].y), weight_shift1));      // Form a GF31 from each pair of input words
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }

  fft_WIDTH(lds, u, smallTrig);
  writeCarryFusedLine(u, out, g);
  bar();
  fft_WIDTH(lds31, u31, smallTrig31);
  writeCarryFusedLine(u31, out31, g);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3231

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig, BigTabFP32 THREAD_WEIGHTS) {
  local F2 ldsF2[WIDTH / 2];
  local GF31 *lds31 = (local GF31 *) ldsF2;
  F2 uF2[NW];
  GF31 u31[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);

  in += g * WIDTH;

  F base = optionalHalve(fancyMul(THREAD_WEIGHTS[me].y, THREAD_WEIGHTS[G_W + g].y));

  u32 word_index = (me * BIG_HEIGHT + g) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 31.
  const u32 log2_root_two = (u32) (((1ULL << 30) / NWORDS) % 31);
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 30) % 31;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits           combo.a[0]
#define weight_shift        combo.a[1]
#define combo_counter       combo.b

  const u64 combo_step = ((u64) bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 combo_bigstep = ((G_W * BIG_HEIGHT * 2 - 1) * combo_step + (((u64) (G_W * BIG_HEIGHT * 2 - 1) * FRAC_BPW_LO) >> 32)) % (31ULL << 32);
  combo_counter = word_index * combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  weight_shift = weight_shift % 31;

  for (u32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // Generate the FP32 weights and the second GF31 weight shift
    F w1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    F w2 = optionalHalve(fancyMul(w1, WEIGHT_STEP));
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 31) weight_shift -= 31;
    u32 weight_shift1 = weight_shift;
    // Convert and weight input
    uF2[i] = U2(in[p].x * w1, in[p].y * w2);
    u31[i] = U2(shl(make_Z31(in[p].x), weight_shift0), shl(make_Z31(in[p].y), weight_shift1));      // Form a GF31 from each pair of input words
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 31) weight_shift -= 31;
  }

  fft_WIDTH(ldsF2, uF2, smallTrigF2);
  writeCarryFusedLine(uF2, outF2, g);
  bar();
  fft_WIDTH(lds31, u31, smallTrig31);
  writeCarryFusedLine(u31, out31, g);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M61^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3261

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig, BigTabFP32 THREAD_WEIGHTS) {
  local GF61 lds61[WIDTH / 2];
  local F2 *ldsF2 = (local F2 *) lds61;
  F2 uF2[NW];
  GF61 u61[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTWTRIGGF61);

  in += g * WIDTH;

  F base = optionalHalve(fancyMul(THREAD_WEIGHTS[me].y, THREAD_WEIGHTS[G_W + g].y));

  u32 word_index = (me * BIG_HEIGHT + g) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 61.
  const u32 log2_root_two = (u32) (((1ULL << 60) / NWORDS) % 61);
  const u32 bigword_weight_shift = (NWORDS - EXP % NWORDS) * log2_root_two % 61;
  const u32 bigword_weight_shift_minus1 = (bigword_weight_shift + 60) % 61;

  // Derive the big vs. little flags from the fractional number of bits in each word.
  // Create a 64-bit counter that tracks both weight shifts and frac_bits (adding 0xFFFFFFFF to effect the ceil operation required for weight shift).
  union { uint2 a; u64 b; } combo;
#define frac_bits           combo.a[0]
#define weight_shift        combo.a[1]
#define combo_counter       combo.b

  const u64 combo_step = ((u64) bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 combo_bigstep = ((G_W * BIG_HEIGHT * 2 - 1) * combo_step + (((u64) (G_W * BIG_HEIGHT * 2 - 1) * FRAC_BPW_LO) >> 32)) % (61ULL << 32);
  combo_counter = word_index * combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  weight_shift = weight_shift % 61;

  for (u32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // Generate the FP32 weights and the second GF61 weight shift
    F w1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    F w2 = optionalHalve(fancyMul(w1, WEIGHT_STEP));
    u32 weight_shift0 = weight_shift;
    combo_counter += combo_step;
    if (weight_shift > 61) weight_shift -= 61;
    u32 weight_shift1 = weight_shift;
    // Convert and weight input
    uF2[i] = U2(in[p].x * w1, in[p].y * w2);
    u61[i] = U2(shl(make_Z61(in[p].x), weight_shift0), shl(make_Z61(in[p].y), weight_shift1));      // Form a GF61 from each pair of input words
    // Generate weight shifts and frac_bits for next pair
    combo_counter += combo_bigstep;
    if (weight_shift > 61) weight_shift -= 61;
  }

  fft_WIDTH(ldsF2, uF2, smallTrigF2);
  writeCarryFusedLine(uF2, outF2, g);
  bar();
  fft_WIDTH(lds61, u61, smallTrig61);
  writeCarryFusedLine(u61, out61, g);
}


/**************************************************************************/
/*    Similar to above, but for an NTT based on GF(M31^2)*GF(M61^2)       */
/**************************************************************************/

#elif FFT_TYPE == FFT3161

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig) {
  local GF61 lds61[WIDTH / 2];
  local GF31 *lds31 = (local GF31 *) lds61;
  GF31 u31[NW];
  GF61 u61[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTWTRIGGF61);

  in += g * WIDTH;

  u32 word_index = (me * BIG_HEIGHT + g) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Weights can be applied with shifts because 2 is the 60th root GF61.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 61.
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
  const u64 m31_combo_bigstep = ((G_W * BIG_HEIGHT * 2 - 1) * m31_combo_step + (((u64) (G_W * BIG_HEIGHT * 2 - 1) * FRAC_BPW_LO) >> 32)) % (31ULL << 32);
  m31_combo_counter = word_index * m31_combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  m31_weight_shift = m31_weight_shift % 31;
  const u64 m61_combo_step = ((u64) m61_bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 m61_combo_bigstep = ((G_W * BIG_HEIGHT * 2 - 1) * m61_combo_step + (((u64) (G_W * BIG_HEIGHT * 2 - 1) * FRAC_BPW_LO) >> 32)) % (61ULL << 32);
  m61_combo_counter = word_index * m61_combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  m61_weight_shift = m61_weight_shift % 61;

  for (u32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // Generate the second weight shifts
    u32 m31_weight_shift0 = m31_weight_shift;
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    u32 m31_weight_shift1 = m31_weight_shift;
    u32 m61_weight_shift0 = m61_weight_shift;
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
    u32 m61_weight_shift1 = m61_weight_shift;
    // Convert and weight input
    u31[i] = U2(shl(make_Z31(in[p].x), m31_weight_shift0), shl(make_Z31(in[p].y), m31_weight_shift1));      // Form a GF31 from each pair of input words
    u61[i] = U2(shl(make_Z61(in[p].x), m61_weight_shift0), shl(make_Z61(in[p].y), m61_weight_shift1));      // Form a GF61 from each pair of input words

// Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_bigstep;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_bigstep;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }

  fft_WIDTH(lds31, u31, smallTrig31);
  writeCarryFusedLine(u31, out31, g);
  bar();
  fft_WIDTH(lds61, u61, smallTrig61);
  writeCarryFusedLine(u61, out61, g);
}


/******************************************************************************/
/*  Similar to above, but for a hybrid FFT based on FP32*GF(M31^2)*GF(M61^2)  */
/******************************************************************************/

#elif FFT_TYPE == FFT323161

// fftPremul: weight words with IBDWT weights followed by FFT-width.
KERNEL(G_W) fftP(P(T2) out, CP(Word2) in, Trig smallTrig, BigTabFP32 THREAD_WEIGHTS) {
  local GF61 lds61[WIDTH / 2];
  local F2 *ldsF2 = (local F2 *) lds61;
  local GF31 *lds31 = (local GF31 *) lds61;
  F2 uF2[NW];
  GF31 u31[NW];
  GF61 u61[NW];

  u32 g = get_group_id(0);
  u32 me = get_local_id(0);

  P(F2) outF2 = (P(F2)) out;
  TrigFP32 smallTrigF2 = (TrigFP32) smallTrig;
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 smallTrig61 = (TrigGF61) (smallTrig + DISTWTRIGGF61);

  in += g * WIDTH;

  F base = optionalHalve(fancyMul(THREAD_WEIGHTS[me].y, THREAD_WEIGHTS[G_W + g].y));

  u32 word_index = (me * BIG_HEIGHT + g) * 2;

  // Weight is 2^[ceil(qj / n) - qj/n] where j is the word index, q is the Mersenne exponent, and n is the number of words.
  // Weights can be applied with shifts because 2 is the 60th root GF61.
  // Let s be the shift amount for word 1.  The shift amount for word x is ceil(x * (s - 1) + num_big_words_less_than_x) % 61.
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
  const u64 m31_combo_bigstep = ((G_W * BIG_HEIGHT * 2 - 1) * m31_combo_step + (((u64) (G_W * BIG_HEIGHT * 2 - 1) * FRAC_BPW_LO) >> 32)) % (31ULL << 32);
  m31_combo_counter = word_index * m31_combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  m31_weight_shift = m31_weight_shift % 31;
  const u64 m61_combo_step = ((u64) m61_bigword_weight_shift_minus1 << 32) + FRAC_BPW_HI;
  const u64 m61_combo_bigstep = ((G_W * BIG_HEIGHT * 2 - 1) * m61_combo_step + (((u64) (G_W * BIG_HEIGHT * 2 - 1) * FRAC_BPW_LO) >> 32)) % (61ULL << 32);
  m61_combo_counter = word_index * m61_combo_step + mul_hi(word_index, FRAC_BPW_LO) + 0xFFFFFFFFULL;
  m61_weight_shift = m61_weight_shift % 61;

  for (u32 i = 0; i < NW; ++i) {
    u32 p = G_W * i + me;
    // Generate the FP32 weights and the second GF31 and GF61 weight shift
    F w1 = i == 0 ? base : optionalHalve(fancyMul(base, fweightStep(i)));
    F w2 = optionalHalve(fancyMul(w1, WEIGHT_STEP));
    u32 m31_weight_shift0 = m31_weight_shift;
    m31_combo_counter += m31_combo_step;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    u32 m31_weight_shift1 = m31_weight_shift;
    u32 m61_weight_shift0 = m61_weight_shift;
    m61_combo_counter += m61_combo_step;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
    u32 m61_weight_shift1 = m61_weight_shift;
    // Convert and weight input
    uF2[i] = U2(in[p].x * w1, in[p].y * w2);
    u31[i] = U2(shl(make_Z31(in[p].x), m31_weight_shift0), shl(make_Z31(in[p].y), m31_weight_shift1));      // Form a GF31 from each pair of input words
    u61[i] = U2(shl(make_Z61(in[p].x), m61_weight_shift0), shl(make_Z61(in[p].y), m61_weight_shift1));      // Form a GF61 from each pair of input words

// Generate weight shifts and frac_bits for next pair
    m31_combo_counter += m31_combo_bigstep;
    m31_weight_shift = adjust_m31_weight_shift(m31_weight_shift);
    m61_combo_counter += m61_combo_bigstep;
    m61_weight_shift = adjust_m61_weight_shift(m61_weight_shift);
  }

  fft_WIDTH(ldsF2, uF2, smallTrigF2);
  writeCarryFusedLine(uF2, outF2, g);
  bar();
  fft_WIDTH(lds31, u31, smallTrig31);
  writeCarryFusedLine(u31, out31, g);
  bar();
  fft_WIDTH(lds61, u61, smallTrig61);
  writeCarryFusedLine(u61, out61, g);
}


#else
error - missing FFTp kernel implementation
#endif

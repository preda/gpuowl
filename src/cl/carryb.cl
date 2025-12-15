// Copyright (C) Mihai Preda

#include "carryutil.cl"

KERNEL(G_W) carryB(P(Word2) io, CP(CarryABM) carryIn) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;
  u32 H = BIG_HEIGHT;

  // Derive the big vs. little flags from the fractional number of bits in each FFT word rather read the flags from memory.
  // Calculate the most significant 32-bits of FRAC_BPW * the index of the FFT word.  Also add FRAC_BPW_HI to test first biglit flag.
  u32 line = gy * CARRY_LEN;
  u32 fft_word_index = (gx * G_W * H + me * H + line) * 2;
  u32 frac_bits = fft_word_index * FRAC_BPW_HI + mad_hi (fft_word_index, FRAC_BPW_LO, FRAC_BPW_HI);

  io += G_W * gx + WIDTH * CARRY_LEN * gy;

  u32 HB = BIG_HEIGHT / CARRY_LEN;

  u32 prev = (gy + HB * G_W * gx + HB * me + (HB * WIDTH - 1)) % (HB * WIDTH);
  u32 prevLine = prev % HB;
  u32 prevCol  = prev / HB;

  CarryABM carry = carryIn[WIDTH * prevLine + prevCol];

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = i * WIDTH + me;
    bool biglit0 = frac_bits + (2*i) * FRAC_BPW_HI <= FRAC_BPW_HI;
    bool biglit1 = frac_bits + (2*i) * FRAC_BPW_HI >= -FRAC_BPW_HI;   // Same as frac_bits + (2*i) * FRAC_BPW_HI + FRAC_BPW_HI <= FRAC_BPW_HI;
    io[p] = carryWord(io[p], &carry, biglit0, biglit1);
    if (!carry) { return; }
  }
}

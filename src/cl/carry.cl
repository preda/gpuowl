// Copyright (C) Mihai Preda

#include "carryutil.cl"

// Carry propagation with optional MUL-3, over CARRY_LEN words.
// Input arrives conjugated and inverse-weighted.

KERNEL(G_W) carry(u32 posROE, P(Word2) out, CP(T2) in, P(CarryABM) carryOut, CP(u32) bits, P(uint) ROE,
                  BigTab THREAD_WEIGHTS, BigTab CARRY_WEIGHTS) {
  u32 g  = get_group_id(0);
  u32 me = get_local_id(0);
  u32 gx = g % NW;
  u32 gy = g / NW;

  CarryABM carry = 0;  
  float roundMax = 0;
  float carryMax = 0;

  // Split 32 bits into CARRY_LEN groups of 2 bits.
#define GPW (16 / CARRY_LEN)
  u32 b = bits[(G_W * g + me) / GPW] >> (me % GPW * (2 * CARRY_LEN));
#undef GPW

  T base = optionalDouble(fancyMul(CARRY_WEIGHTS[gy].x, THREAD_WEIGHTS[me].x));
  
    base = optionalDouble(fancyMul(base, iweightStep(gx)));

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;
    double w1 = i == 0 ? base : optionalDouble(fancyMul(base, iweightUnitStep(i)));
    double w2 = optionalDouble(fancyMul(w1, IWEIGHT_STEP));
    T2 x = conjugate(in[p]) * U2(w1, w2);
        
#if DO_MUL3
    out[p] = carryPairMul(x, &carry, test(b, 2 * i), test(b, 2 * i + 1), carry, &roundMax, &carryMax);
#else
    out[p] = carryPair(x, &carry, test(b, 2 * i), test(b, 2 * i + 1), carry, &roundMax, &carryMax);
#endif
  }
  carryOut[G_W * g + me] = carry;

#if STATS & (1 << (2 + DO_MUL3))
#if STATS & 16
  updateStats(ROE, posROE, carryMax);
#else
  updateStats(ROE, posROE, roundMax);
#endif
#endif
}

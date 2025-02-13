// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"

#if STATS || ROE
void updateStats(global uint *bufROE, u32 posROE, float roundMax) {
  assert(roundMax >= 0);
  // work_group_reduce_max() allocates an additional 256Bytes LDS for a 64lane workgroup, so avoid it.
  // u32 groupRound = work_group_reduce_max(as_uint(roundMax));
  // if (get_local_id(0) == 0) { atomic_max(bufROE + posROE, groupRound); }

  // Do the reduction directly over global mem.
  atomic_max(bufROE + posROE, as_uint(roundMax));
}
#endif

#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_sbfe)
i32 lowBits(i32 u, u32 bits) { return __builtin_amdgcn_sbfe(u, 0, bits); }
#else
i32 lowBits(i32 u, u32 bits) { return ((u << (32 - bits)) >> (32 - bits)); }
#endif

#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_ubfe)
i32 ulowBits(i32 u, u32 bits) { return __builtin_amdgcn_ubfe(u, 0, bits); }
#else
i32 ulowBits(i32 u, u32 bits) { u32 uu = (u32) u; return ((uu << (32 - bits)) >> (32 - bits)); }
#endif

#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_alignbit)
i32 xtract32(i64 x, u32 bits) { return __builtin_amdgcn_alignbit(as_int2(x).y, as_int2(x).x, bits); }
#else
i32 xtract32(i64 x, u32 bits) { return x >> bits; }
#endif

#if !defined(LL)
#define LL 0
#endif

u32 bitlen(bool b) { return EXP / NWORDS + b; }
bool test(u32 bits, u32 pos) { return (bits >> pos) & 1; }

#if 0
// Check for round off errors above a threshold (default is 0.43)
void ROUNDOFF_CHECK(double x) {
#if DEBUG
#ifndef ROUNDOFF_LIMIT
#define ROUNDOFF_LIMIT 0.43
#endif
  float error = fabs(x - rint(x));
  if (error > ROUNDOFF_LIMIT) printf("Roundoff: %g %30.2f\n", error, x);
#endif
}
#endif

// Rounding constant: 3 * 2^51, See https://stackoverflow.com/questions/17035464
#define RNDVAL (3.0 * (1l << 51))

// Convert a double to long efficiently.  Double must be in RNDVAL+integer format.
i64 RNDVALdoubleToLong(double d) {
  int2 words = as_int2(d);
#if EXP / NWORDS >= 19
  // We extend the range to 52 bits instead of 51 by taking the sign from the negation of bit 51
  words.y ^= 0x00080000u;
  words.y = lowBits(words.y, 20);
#else
  // Take the sign from bit 50 (i.e. use lower 51 bits).
  words.y = lowBits(words.y, 19);
#endif
  return as_long(words);
}

// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i64 weightAndCarryOne(T u, T invWeight, i64 inCarry, float* maxROE, int sloppy_result_is_acceptable) {

#if !MUL3

  // Convert carry into RNDVAL + carry.
  int2 tmp = as_int2(inCarry); tmp.y += as_int2(RNDVAL).y;
  double RNDVALCarry = as_double(tmp);

  // Apply inverse weight and RNDVAL+carry
  double d = fma(u, invWeight, RNDVALCarry);

  // Optionally calculate roundoff error
  float roundoff = fabs((float) fma(u, -invWeight, d - RNDVALCarry));
  *maxROE = max(*maxROE, roundoff);

  // Convert to long (for CARRY32 case we don't need to strip off the RNDVAL bits)
  if (sloppy_result_is_acceptable) return as_long(d);
  else return RNDVALdoubleToLong(d);

#else  // We cannot add in the carry until after the mul by 3

  // Apply inverse weight and RNDVAL
  double d = fma(u, invWeight, RNDVAL);

  // Optionally calculate roundoff error
  float roundoff = fabs((float) fma(u, -invWeight, d - RNDVAL));
  *maxROE = max(*maxROE, roundoff);

  // Convert to long, mul by 3, and add carry
  return RNDVALdoubleToLong(d) * 3 + inCarry;

#endif
}

Word OVERLOAD carryStep(i64 x, i64 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);
  x -= w;
  *outCarry = x >> nBits;
  return w;
}

Word OVERLOAD carryStep(i64 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);
  *outCarry = xtract32(x, nBits) + (w < 0);
  return w;
}

Word OVERLOAD carryStep(i32 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);
  *outCarry = (x - w) >> nBits;
  return w;
}

Word OVERLOAD carryStepSloppy(i64 x, i64 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = ulowBits(x, nBits);
  *outCarry = x >> nBits;
  return w;
}

Word OVERLOAD carryStepSloppy(i64 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = ulowBits(x, nBits);
  *outCarry = xtract32(x, nBits);
  return w;
}

Word OVERLOAD carryStepSloppy(i32 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = ulowBits(x, nBits);
  *outCarry = x >> nBits;
  return w;
}

// map abs(carry) to floats, with 2^32 corresponding to 1.0
// So that the maximum CARRY32 abs(carry), 2^31, is mapped to 0.5 (the same as the maximum ROE)
float OVERLOAD boundCarry(i32 c) { return ldexp(fabs((float) c), -32); }
float OVERLOAD boundCarry(i64 c) { return ldexp(fabs((float) (i32) (c >> 8)), -24); }

#define iCARRY i32
#include "carryinc.cl"
#undef iCARRY

#define iCARRY i64
#include "carryinc.cl"
#undef iCARRY

#if CARRY64
typedef i64 CFcarry;
#else
typedef i32 CFcarry;
#endif

// The carry for the non-fused CarryA, CarryB, CarryM kernels.
// Simply use large carry always as the split kernels are slow anyway (and seldomly used normally).
typedef i64 CarryABM;

// Carry propagation from word and carry.
Word2 carryWord(Word2 a, CarryABM* carry, bool b1, bool b2) {
  a.x = carryStep(a.x + *carry, carry, b1);
  a.y = carryStep(a.y + *carry, carry, b2);
  return a;
}


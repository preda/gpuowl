// Copyright (C) Mihai Preda

#include "base.cl"
#include "math.cl"

#if STATS
void updateStats(global uint *ROE, u32 posROE, float roundMax) {
  assert(roundMax >= 0);
  u32 groupRound = work_group_reduce_max(as_uint(roundMax));

  if (get_local_id(0) == 0) { atomic_max(ROE + posROE, groupRound); }
}
#endif

#if 0 && HAS_ASM
i32  lowBits(i32 u, u32 bits) { i32 tmp; __asm("v_bfe_i32 %0, %1, 0, %2" : "=v" (tmp) : "v" (u), "v" (bits)); return tmp; }
i32 xtract32(i64 x, u32 bits) { i32 tmp; __asm("v_alignbit_b32 %0, %1, %2, %3" : "=v"(tmp) : "v"(as_int2(x).y), "v"(as_int2(x).x), "v"(bits)); return tmp; }
#else
i32  lowBits(i32 u, u32 bits) { return ((u << (32 - bits)) >> (32 - bits)); }
i32 xtract32(i64 x, u32 bits) { return x >> bits; }
#endif

#if !defined(LL)
#define LL 0
#endif

u32 bitlen(bool b) { return EXP / NWORDS + b; }
bool test(u32 bits, u32 pos) { return (bits >> pos) & 1; }

// We support two sizes of carry in carryFused.  A 32-bit carry halves the amount of memory used by CarryShuttle,
// but has some risks.  As FFT sizes increase and/or exponents approach the limit of an FFT size, there is a chance
// that the carry will not fit in 32-bits -- corrupting results.  That said, I did test 2000 iterations of an exponent
// just over 1 billion.  Max(abs(carry)) was 0x637225E9 which is OK (0x80000000 or more is fatal).  P-1 testing is more
// problematic as the mul-by-3 triples the carry too.

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

// Rounding constant: 3 * 2^51, See https://stackoverflow.com/questions/17035464
#define RNDVAL (3.0 * (1l << 51))

i64 doubleToLong(double x, float* maxROE) {
  // Unfortunatelly (i64) rint() is slow!
  // return rint(x);

  ROUNDOFF_CHECK(x);

  double d = x + RNDVAL;
  float roundoff = fabs((float) (x - (d - RNDVAL)));
  *maxROE = max(*maxROE, roundoff);

  // i32 roundoff = abs(as_int2(x - (d - RNDVAL2)).x);

  int2 words = as_int2(d);

#if EXP / NWORDS >= 19
  // We extend the range to 52 bits instead of 51 by taking the sign from the negation of bit 51
  words.y ^= 0x00080000u;
  words.y = lowBits(words.y, 20);

#if 0
  words.y <<= 12;
  words.y ^= 0x80000000u;
  words.y >>= 12;
#endif
#else
  // Take the sign from bit 50 (i.e. use lower 51 bits).
  words.y = lowBits(words.y, 19);
#endif

  return as_long(words);
}

Word OVERLOAD carryStep(i64 x, i64 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);
  x -= w;
  *outCarry = x >> nBits;
  return w;
}

// Check for 32-bit carry nearing the limit (default is 0x7C000000)
void CARRY32_CHECK(i32 x) {
#if DEBUG
#ifndef CARRY32_LIMIT
#define CARRY32_LIMIT 0x7C000000
#endif
  if (abs(x) > CARRY32_LIMIT) { printf("Carry32: %X\n", abs(x)); }
#endif
}

Word OVERLOAD carryStep(i64 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);
  *outCarry = xtract32(x, nBits) + (w < 0);
  CARRY32_CHECK(*outCarry);
  return w;
}

Word OVERLOAD carryStep(i32 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);
  *outCarry = (x - w) >> nBits;
  CARRY32_CHECK(*outCarry);
  return w;
}

// map abs(carry) to floats, with 2^32 corresponding to 1.0
float OVERLOAD boundCarry(i32 c) { return ldexp(fabs((float) c), -32); }
float OVERLOAD boundCarry(i64 c) { return ldexp(fabs((float) (i32) (c >> 8)), -24); }

#define iCARRY i32
#include "carryinc.cl"
#undef iCARRY

#define iCARRY i64
#include "carryinc.cl"
#undef iCARRY

// In the carryMul situation there's an additional multiplication by 3, adding about 1.6bits to carry, so 32bits
// is often not enough.
typedef i64 CFMcarry;

#if CARRY32
typedef i32 CFcarry;
#else
typedef i64 CFcarry;
#endif

Word2 OVERLOAD carryPairMul(T2 u, i64 *outCarry, bool b1, bool b2, i64 inCarry, float* maxROE, float* carryMax) {
  i64 midCarry;
  Word a = carryStep(3 * doubleToLong(u.x, maxROE) + inCarry, &midCarry, b1);
  Word b = carryStep(3 * doubleToLong(u.y, maxROE) + midCarry, outCarry, b2);
// #if STATS & 0xA
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
// #endif
  return (Word2) (a, b);
}

// Carry propagation from word and carry.
Word2 carryWord(Word2 a, CarryABM* carry, bool b1, bool b2) {
  a.x = carryStep(a.x + *carry, carry, b1);
  a.y = carryStep(a.y + *carry, carry, b2);
  return a;
}


double optionalDouble(double iw) {
  assert(iw > 0.5 && iw < 2);
  uint2 u = as_uint2(iw);

  u.y |= 0x00100000;
  return as_double(u);
}

double optionalHalve(double w) {    // return w >= 4 ? w / 2 : w;
  assert(w >= 2 && w < 8);
  uint2 u = as_uint2(w);
  // u.y &= 0xFFEFFFFF;
  u.y = bfi(u.y, 0xffefffff, 0);
  return as_double(u);
}

// Rounding constant: 3 * 2^51, See https://stackoverflow.com/questions/17035464
#define RNDVAL (3.0 * (1l << 51))
// Top 32-bits of RNDVAL
#define TOPVAL (0x43380000)

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

// We support two sizes of carry in carryFused.  A 32-bit carry halves the amount of memory used by CarryShuttle,
// but has some risks.  As FFT sizes increase and/or exponents approach the limit of an FFT size, there is a chance
// that the carry will not fit in 32-bits -- corrupting results.  That said, I did test 2000 iterations of an exponent
// just over 1 billion.  Max(abs(carry)) was 0x637225E9 which is OK (0x80000000 or more is fatal).  P-1 testing is more
// problematic as the mul-by-3 triples the carry too.

// For CARRY32 we don't mind pollution of this value with the double exponent bits
i64 OVERLOAD doubleToLong(double x, i32 inCarry) {
  ROUNDOFF_CHECK(x);
  return as_long(x + as_double((int2) (inCarry, TOPVAL - (inCarry < 0))));
}

i64 OVERLOAD doubleToLong(double x, i64 inCarry) {
  ROUNDOFF_CHECK(x);
  int2 tmp = as_int2(inCarry);
  tmp.y += TOPVAL;
  double d = x + as_double(tmp);

  // Extend the sign from the lower 51-bits.
  // The first 51 bits (0 to 50) are clean. Bit 51 is affected by RNDVAL. Bit 52 of RNDVAL is not stored.
  // Note: if needed, we can extend the range to 52 bits instead of 51 by taking the sign from the negation
  // of bit 51 (i.e. bit 51==1 means positive, bit 51==0 means negative).
  int2 data = as_int2(d);
  data.y = lowBits(data.y, 51 - 32);
  return as_long(data);
}

Word OVERLOAD carryStep(i64 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);

// If nBits could be 20 or more we must be careful.  doubleToLong generated x as 13 bits of trash and 51-bit signed value.
// If we right shift 20 bits we will shift some of the trash into outCarry.  First we must remove the trash bits.
#if EXP / NWORDS >= 19
  *outCarry = as_int2(x << 13).y >> (nBits - 19);
#else
  *outCarry = xtract32(x, nBits);
#endif
  *outCarry += (w < 0);
  CARRY32_CHECK(*outCarry);
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

Word OVERLOAD carryStep(i32 x, i32 *outCarry, bool isBigWord) {
  u32 nBits = bitlen(isBigWord);
  Word w = lowBits(x, nBits);
  *outCarry = (x - w) >> nBits;
  CARRY32_CHECK(*outCarry);
  return w;
}

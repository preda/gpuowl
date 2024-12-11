// Copyright (C) Mihai Preda

// This file is included with different definitions for iCARRY

Word2 OVERLOAD carryPair(long2 u, iCARRY *outCarry, bool b1, bool b2, float* carryMax) {
  iCARRY midCarry;
  Word a = carryStep(u.x, &midCarry, b1);
  Word b = carryStep(u.y + midCarry, outCarry, b2);
// #if STATS & 0x5
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
// #endif
  return (Word2) (a, b);
}

Word2 OVERLOAD carryFinal(Word2 u, iCARRY inCarry, bool b1) {
  i32 tmpCarry;
  u.x = carryStep(u.x + inCarry, &tmpCarry, b1);
  u.y += tmpCarry;
  return u;
}


// Apply inverse weight, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
i64 OVERLOAD weightAndCarryOne(T u, T invWeight, iCARRY inCarry, float* maxROE) {

#if !MUL3

  // Convert carry into RNDVAL + carry.
  double RNDVALCarry = RNDVAL;
  RNDVALCarry = as_double(as_long(RNDVALCarry) + inCarry);

  // Apply inverse weight and RNDVAL+carry
  double d = fma(u, invWeight, RNDVALCarry);

  // Optionally calculate roundoff error
  float roundoff = fabs((float) fma(u, -invWeight, d - RNDVALCarry));
  *maxROE = max(*maxROE, roundoff);

  // Convert to long
  return RNDVALdoubleToLong(d);

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

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
long2 OVERLOAD weightAndCarry(T2 u, T2 invWeight, iCARRY inCarry, float* maxROE) {
  return (long2) (weightAndCarryOne(u.x, invWeight.x, inCarry, maxROE), weightAndCarryOne(u.y, invWeight.y, 0, maxROE));
}

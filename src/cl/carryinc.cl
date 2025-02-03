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


// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(T2 u, T2 invWeight, i64 inCarry, float* maxROE, iCARRY *outCarry, bool b1, bool b2, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(u.x, invWeight.x, inCarry, maxROE, sizeof(midCarry) == 4);
  Word a = carryStep(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(u.y, invWeight.y, midCarry, maxROE, sizeof(midCarry) == 4);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accuracy calculation of the first carry is not required.
Word2 OVERLOAD weightAndCarryPairSloppy(T2 u, T2 invWeight, i64 inCarry, float* maxROE, iCARRY *outCarry, bool b1, bool b2, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(u.x, invWeight.x, inCarry, maxROE, sizeof(midCarry) == 4);
  Word a = carryStepSloppy(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(u.y, invWeight.y, midCarry, maxROE, sizeof(midCarry) == 4);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

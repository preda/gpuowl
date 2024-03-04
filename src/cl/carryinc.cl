// Copyright (C) Mihai Preda

// This file is included with different definitions for iCarry

Word2 OVERLOAD carryPair(T2 u, iCARRY *outCarry, bool b1, bool b2, iCARRY inCarry, float *maxROE, float* carryMax) {
  iCARRY midCarry;
  Word a = carryStep(doubleToLong(u.x, maxROE) + inCarry, &midCarry, b1);
  Word b = carryStep(doubleToLong(u.y, maxROE) + midCarry, outCarry, b2);
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

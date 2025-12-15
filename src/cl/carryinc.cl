// Copyright (C) Mihai Preda

// This file is included with different definitions for iCARRY

Word2 OVERLOAD carryFinal(Word2 u, iCARRY inCarry, bool b1) {
  i32 tmpCarry;
  u.x = carryStepSignedSloppy(u.x + inCarry, &tmpCarry, b1);
  u.y += tmpCarry;
  return u;
}

/*******************************************************************************************/
/*  Original FP64 version to start the carry propagation process for a pair of FFT values  */
/*******************************************************************************************/

#if FFT_TYPE == FFT64

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(T2 u, T2 invWeight, i64 inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(u.x, invWeight.x, inCarry, maxROE, sizeof(midCarry) == 4);
  Word a = carryStep(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(u.y, invWeight.y, midCarry, maxROE, sizeof(midCarry) == 4);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(T2 u, T2 invWeight, i64 inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(u.x, invWeight.x, inCarry, maxROE, sizeof(midCarry) == 4);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(u.y, invWeight.y, midCarry, maxROE, sizeof(midCarry) == 4);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#elif FFT_TYPE == FFT32

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(F2 u, F2 invWeight, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  i32 midCarry;
  i32 tmp1 = weightAndCarryOne(u.x, invWeight.x, inCarry, maxROE, sizeof(midCarry) == 4);
  Word a = carryStep(tmp1, &midCarry, b1);
  i32 tmp2 = weightAndCarryOne(u.y, invWeight.y, midCarry, maxROE, sizeof(midCarry) == 4);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(F2 u, F2 invWeight, iCARRY inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  i32 midCarry;
  i32 tmp1 = weightAndCarryOne(u.x, invWeight.x, inCarry, maxROE, sizeof(midCarry) == 4);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i32 tmp2 = weightAndCarryOne(u.y, invWeight.y, midCarry, maxROE, sizeof(midCarry) == 4);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT31

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(GF31 u, u32 invWeight1, u32 invWeight2, i32 inCarry, bool b1, bool b2, iCARRY *outCarry, u32* maxROE, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(u.x, invWeight1, inCarry, maxROE);
  Word a = carryStep(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(u.y, invWeight2, midCarry, maxROE);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(GF31 u, u32 invWeight1, u32 invWeight2, i32 inCarry, bool b1, bool b2, iCARRY *outCarry, u32* maxROE, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(u.x, invWeight1, inCarry, maxROE);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(u.y, invWeight2, midCarry, maxROE);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#elif FFT_TYPE == FFT61

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(GF61 u, u32 invWeight1, u32 invWeight2, i64 inCarry, bool b1, bool b2, iCARRY *outCarry, u32* maxROE, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(u.x, invWeight1, inCarry, maxROE);
  Word a = carryStep(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(u.y, invWeight2, midCarry, maxROE);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(GF61 u, u32 invWeight1, u32 invWeight2, i64 inCarry, bool b1, bool b2, iCARRY *outCarry, u32* maxROE, float* carryMax) {
  iCARRY midCarry;
  i64 tmp1 = weightAndCarryOne(u.x, invWeight1, inCarry, maxROE);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(u.y, invWeight2, midCarry, maxROE);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP64 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT6431

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(T2 u, GF31 u31, T invWeight1, T invWeight2, u32 m31_invWeight1, u32 m31_invWeight2,
                                  bool hasInCarry, i64 inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  i64 midCarry;
  i96 tmp1 = weightAndCarryOne(u.x, u31.x, invWeight1, m31_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStep(tmp1, &midCarry, b1);
  i96 tmp2 = weightAndCarryOne(u.y, u31.y, invWeight2, m31_invWeight2, true, midCarry, maxROE);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(T2 u, GF31 u31, T invWeight1, T invWeight2, u32 m31_invWeight1, u32 m31_invWeight2,
                                        bool hasInCarry, i64 inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  i64 midCarry;
  i96 tmp1 = weightAndCarryOne(u.x, u31.x, invWeight1, m31_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i96 tmp2 = weightAndCarryOne(u.y, u31.y, invWeight2, m31_invWeight2, true, midCarry, maxROE);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M31^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3231

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(F2 uF2, GF31 u31, F invWeight1, F invWeight2, u32 m31_invWeight1, u32 m31_invWeight2,
                                  i32 inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  i32 midCarry;
  i64 tmp1 = weightAndCarryOne(uF2.x, u31.x, invWeight1, m31_invWeight1, inCarry, maxROE);
  Word a = carryStep(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(uF2.y, u31.y, invWeight2, m31_invWeight2, midCarry, maxROE);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(F2 uF2, GF31 u31, F invWeight1, F invWeight2, u32 m31_invWeight1, u32 m31_invWeight2,
                                        i32 inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  i32 midCarry;
  i64 tmp1 = weightAndCarryOne(uF2.x, u31.x, invWeight1, m31_invWeight1, inCarry, maxROE);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i64 tmp2 = weightAndCarryOne(uF2.y, u31.y, invWeight2, m31_invWeight2, midCarry, maxROE);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}


/**************************************************************************/
/*    Similar to above, but for a hybrid FFT based on FP32 & GF(M61^2)    */
/**************************************************************************/

#elif FFT_TYPE == FFT3261

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(F2 uF2, GF61 u61, F invWeight1, F invWeight2, u32 m61_invWeight1, u32 m61_invWeight2,
                                  bool hasInCarry, i64 inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  i64 midCarry;
  i96 tmp1 = weightAndCarryOne(uF2.x, u61.x, invWeight1, m61_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStep(tmp1, &midCarry, b1);
  i96 tmp2 = weightAndCarryOne(uF2.y, u61.y, invWeight2, m61_invWeight2, true, midCarry, maxROE);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(F2 uF2, GF61 u61, F invWeight1, F invWeight2, u32 m61_invWeight1, u32 m61_invWeight2,
                                        bool hasInCarry, i64 inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  i64 midCarry;
  i96 tmp1 = weightAndCarryOne(uF2.x, u61.x, invWeight1, m61_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i96 tmp2 = weightAndCarryOne(uF2.y, u61.y, invWeight2, m61_invWeight2, true, midCarry, maxROE);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}


/**************************************************************************/
/*    Similar to above, but for an NTT based on GF(M31^2)*GF(M61^2)       */
/**************************************************************************/

#elif FFT_TYPE == FFT3161

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(GF31 u31, GF61 u61, u32 m31_invWeight1, u32 m31_invWeight2, u32 m61_invWeight1, u32 m61_invWeight2,
                                  bool hasInCarry, i64 inCarry, bool b1, bool b2, iCARRY *outCarry, u32* maxROE, float* carryMax) {
  iCARRY midCarry;
  i96 tmp1 = weightAndCarryOne(u31.x, u61.x, m31_invWeight1, m61_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStep(tmp1, &midCarry, b1);
  i96 tmp2 = weightAndCarryOne(u31.y, u61.y, m31_invWeight2, m61_invWeight2, true, midCarry, maxROE);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(GF31 u31, GF61 u61, u32 m31_invWeight1, u32 m31_invWeight2, u32 m61_invWeight1, u32 m61_invWeight2,
                                        bool hasInCarry, i64 inCarry, bool b1, bool b2, iCARRY *outCarry, u32* maxROE, float* carryMax) {
  iCARRY midCarry;
  i96 tmp1 = weightAndCarryOne(u31.x, u61.x, m31_invWeight1, m61_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i96 tmp2 = weightAndCarryOne(u31.y, u61.y, m31_invWeight2, m61_invWeight2, true, midCarry, maxROE);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

/******************************************************************************/
/*  Similar to above, but for a hybrid FFT based on FP32*GF(M31^2)*GF(M61^2)  */
/******************************************************************************/

#elif FFT_TYPE == FFT323161

// Apply inverse weights, add in optional carry, calculate roundoff error, convert to integer. Handle MUL3.
// Then propagate carries through two words.  Generate the output carry.
Word2 OVERLOAD weightAndCarryPair(F2 uF2, GF31 u31, GF61 u61, F invWeight1, F invWeight2, u32 m31_invWeight1, u32 m31_invWeight2,
                                  u32 m61_invWeight1, u32 m61_invWeight2, bool hasInCarry, i64 inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i128 tmp1 = weightAndCarryOne(uF2.x, u31.x, u61.x, invWeight1, m31_invWeight1, m61_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStep(tmp1, &midCarry, b1);
  i128 tmp2 = weightAndCarryOne(uF2.y, u31.y, u61.y, invWeight2, m31_invWeight2, m61_invWeight2, true, midCarry, maxROE);
  Word b = carryStep(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

// Like weightAndCarryPair except that a strictly accurate calculation of the first Word and carry is not required.  Second word may also be sloppy.
Word2 OVERLOAD weightAndCarryPairSloppy(F2 uF2, GF31 u31, GF61 u61, F invWeight1, F invWeight2, u32 m31_invWeight1, u32 m31_invWeight2,
                                        u32 m61_invWeight1, u32 m61_invWeight2, bool hasInCarry, i64 inCarry, bool b1, bool b2, iCARRY *outCarry, float* maxROE, float* carryMax) {
  iCARRY midCarry;
  i128 tmp1 = weightAndCarryOne(uF2.x, u31.x, u61.x, invWeight1, m31_invWeight1, m61_invWeight1, hasInCarry, inCarry, maxROE);
  Word a = carryStepUnsignedSloppy(tmp1, &midCarry, b1);
  i128 tmp2 = weightAndCarryOne(uF2.y, u31.y, u61.y, invWeight2, m31_invWeight2, m61_invWeight2, true, midCarry, maxROE);
  Word b = carryStepSignedSloppy(tmp2, outCarry, b2);
  *carryMax = max(*carryMax, max(boundCarry(midCarry), boundCarry(*outCarry)));
  return (Word2) (a, b);
}

#else
error - missing weightAndCarryPair implementation
#endif

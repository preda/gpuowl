// Copyright (C) Mihai Preda

#pragma once

#include "fft4.cl"

#if FFT_FP64

T2 mul_t8_delayed(T2 a) { return U2(a.x - a.y, a.x + a.y); }
T2 mul_3t8_delayed(T2 a) { return U2(-(a.x + a.y), a.x - a.y); }
//#define X2_apply_delay(a, b) { T2 t = a; a = t + M_SQRT1_2 * b; b = t - M_SQRT1_2 * b; }
#define X2_apply_delay(a, b) { T2 t = a; a.x = fma(b.x, M_SQRT1_2, a.x); a.y = fma(b.y, M_SQRT1_2, a.y); b.x = fma(-M_SQRT1_2, b.x, t.x); b.y = fma(-M_SQRT1_2, b.y, t.y); }

void OVERLOAD fft4CoreSpecial(T2 *u) {
  X2(u[0], u[2]);
  X2_mul_t4(u[1], u[3]);                                        // X2(u[1], u[3]); u[3] = mul_t4(u[3]);
  X2_apply_delay(u[0], u[1]);
  X2_apply_delay(u[2], u[3]);
}

void OVERLOAD fft8Core(T2 *u) {
  X2(u[0], u[4]);
  X2(u[1], u[5]);   u[5] = mul_t8_delayed(u[5]);
  X2_mul_t4(u[2], u[6]);                                        // X2(u[2], u[6]);   u[6] = mul_t4(u[6]);
  X2(u[3], u[7]);   u[7] = mul_3t8_delayed(u[7]);
  fft4Core(u);
  fft4CoreSpecial(u + 4);
}

// 4 MUL + 52 ADD
void OVERLOAD fft8(T2 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

F2 mul_t8_delayed(F2 a) { return U2(a.x - a.y, a.x + a.y); }
F2 mul_3t8_delayed(F2 a) { return U2(-(a.x + a.y), a.x - a.y); }
//#define X2_apply_delay(a, b) { F2 t = a; a = t + M_SQRT1_2 * b; b = t - M_SQRT1_2 * b; }
#define X2_apply_delay(a, b) { F2 t = a; a.x = fma(b.x, (float) M_SQRT1_2, a.x); a.y = fma(b.y, (float) M_SQRT1_2, a.y); b.x = fma((float) -M_SQRT1_2, b.x, t.x); b.y = fma((float) -M_SQRT1_2, b.y, t.y); }

void OVERLOAD fft4CoreSpecial(F2 *u) {
  X2(u[0], u[2]);
  X2_mul_t4(u[1], u[3]);                                        // X2(u[1], u[3]); u[3] = mul_t4(u[3]);
  X2_apply_delay(u[0], u[1]);
  X2_apply_delay(u[2], u[3]);
}

void OVERLOAD fft8Core(F2 *u) {
  X2(u[0], u[4]);
  X2(u[1], u[5]);   u[5] = mul_t8_delayed(u[5]);
  X2_mul_t4(u[2], u[6]);                                        // X2(u[2], u[6]);   u[6] = mul_t4(u[6]);
  X2(u[3], u[7]);   u[7] = mul_3t8_delayed(u[7]);
  fft4Core(u);
  fft4CoreSpecial(u + 4);
}

// 4 MUL + 52 ADD
void OVERLOAD fft8(F2 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

void OVERLOAD fft8Core(GF31 *u) {
  X2(u[0], u[4]);
  X2_mul_t8(u[1], u[5]);
  X2_mul_t4(u[2], u[6]);
  X2_mul_3t8(u[3], u[7]);
  fft4Core(u);
  fft4Core(u + 4);
}

// 4 MUL + 52 ADD
void OVERLOAD fft8(GF31 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

#if 0 // Working code.

void OVERLOAD fft8Core(GF61 *u) {
  X2(u[0], u[4]);                                               //GWBUG: Delay some mods using extra 3 bits of Z61
  X2_mul_t8(u[1], u[5]);                                        // X2(u[1], u[5]);   u[5] = mul_t8(u[5]);
  X2_mul_t4(u[2], u[6]);                                        // X2(u[2], u[6]);   u[6] = mul_t4(u[6]);
  X2_mul_3t8(u[3], u[7]);                                       // X2(u[3], u[7]);   u[7] = mul_3t8(u[7]);
  fft4Core(u);
  fft4Core(u + 4);
}

void OVERLOAD fft8(GF61 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#else   // Carefully track the size of numbers to reduce the number of mod M61 reductions

void OVERLOAD fft4CoreSpecial1(GF61 *u) {         // Starts with u[0,1,2,3] having maximum values of (2,2,3,2)*M61+epsilon.
  X2q(&u[0], &u[2], 4);                           // X2(u[0], u[2]);  No reductions mod M61.  u[0,2] max value is 5,6*M61+epsilon.
  X2q_mul_t4(&u[1], &u[3], 3);                    // X2(u[1], u[3]); u[3] = mul_t4(u[3]);  u[1,3] max value is 5,4*M61+epsilon.
  u[1] = modM61(u[1]); u[2] = modM61(u[2]);       // Reduce the worst offenders.  u[0,1,2,3] have maximum values of (5,1,1,4)*M61+epsilon.
  X2s(&u[0], &u[1], 2);                           // u[0,1] max value before reduction is 6,7*M61+epsilon
  X2s(&u[2], &u[3], 5);                           // u[2,3] max value before reduction is 5,6*M61+epsilon
}

void OVERLOAD fft4CoreSpecial2(GF61 *u) {         // Similar to above.  Starts with u[0,1,2,3] having maximum values of (3,1,2,1)*M61+epsilon.
  X2q(&u[0], &u[2], 3);                           // u[0,2] max value is 5,6*M61+epsilon.
  X2q_mul_t4(&u[1], &u[3], 2);                    // X2(u[1], u[3]); u[3] = mul_t4(u[3]);  u[1,3] max value is 3,2*M61+epsilon.
  u[0] = modM61(u[0]); u[2] = modM61(u[2]);       // Reduce the worst offenders  u[0,1,2,3] have maximum values of (1,3,1,2)*M61+epsilon.
  X2s(&u[0], &u[1], 4);                           // u[0,1] max value before reduction is 4,5*M61+epsilon
  X2s(&u[2], &u[3], 3);                           // u[2,3] max value before reduction is 3,4*M61+epsilon
}

void OVERLOAD fft8Core(GF61 *u) {                 // Starts with all u[i] having maximum values of M61+epsilon.
  X2q(&u[0], &u[4], 2);                           // X2(u[0], u[4]);  No reductions mod M61.  u[0,4] max value is 2,3*M61+epsilon.
  X2q_mul_t8(&u[1], &u[5], 2);                    // X2(u[1], u[5]);  u[5] = mul_t8(u[5]); u[1,5] max value is 2,1*M61+epsilon.
  X2q_mul_t4(&u[2], &u[6], 2);                    // X2(u[2], u[6]);  u[6] = mul_t4(u[6]); u[2,6] max value is 3,2*M61+epsilon.
  X2q_mul_3t8(&u[3], &u[7], 2);                   // X2(u[3], u[7]);  u[7] = mul_3t8(u[7]); u[3,7] max value is 2,1*M61+epsilon.
  fft4CoreSpecial1(u);
  fft4CoreSpecial2(u + 4);
}

void OVERLOAD fft8(GF61 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif

#endif

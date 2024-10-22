// Copyright (C) Mihai Preda


#if 0

#if 1

#include "fft4.cl"

// 24 FMA (of which 16 MUL) + 136 ADD
void fft16(T2 *u) {
  double
      C1 = 0.92387953251128674, // cos(tau/16)
      S1 = 0.38268343236508978; // sin(tau/16)

  for (int i = 0; i < 4; ++i) { fft4by(u, i, 4, 16); }

  u[5]  = cmul(u[ 5],  U2(C1, S1));
  u[7]  = cmul(u[ 7],  U2(S1, C1));
  u[13] = cmul(u[13],  U2(S1, C1));
  u[15] = cmul(u[15], -U2(C1, S1));

  u[6]  = mul_t8(u[6]);
  u[9]  = mul_t8(u[9]);
  u[11] = mul_3t8(u[11]);
  u[14] = mul_3t8(u[14]);

  u[10] = mul_t4(u[10]);

  for (int i = 0; i < 4; ++i) { fft4by(u, 4 * i, 1, 16); }

  SWAP(u[1], u[4]);
  SWAP(u[2], u[8]);
  SWAP(u[3], u[12]);
  SWAP(u[6], u[9]);
  SWAP(u[7], u[13]);
  SWAP(u[11], u[14]);

  // for (int i = 0; i < 4; ++i) { fft4by(u, i, 4, 16); }
}

#else

#include "fft8.cl"

void fft16(T2 *u) {
  double
      C1 = 0.92387953251128674, // cos(tau/16)
      S1 = 0.38268343236508978; // sin(tau/16)

  for (int i = 0; i < 8; ++i) { X2(u[i], u[i + 8]); }
  u[ 9] = cmul(u[ 9], U2( C1, S1)); // 1t16
  u[11] = cmul(u[11], U2( S1, C1)); // 3t16
  u[13] = cmul(u[13], U2(-S1, C1)); // 5t16
  u[15] = cmul(u[15], U2(-C1, S1)); // 7t16

  u[10] = mul_t8(u[10]);
  u[14] = mul_3t8(u[14]);

  u[12] = mul_t4(u[12]);

  fft8Core(u);
  fft8Core(u + 8);

  // revbin fix order
  // 0 8 4 12 2 10 6 14 1 9 5 13 3 11 7 15
  SWAP(u[1], u[8]);
  SWAP(u[2], u[4]);
  SWAP(u[3], u[12]);
  SWAP(u[5], u[10]);
  SWAP(u[7], u[14]);
  SWAP(u[11], u[13]);
}
#endif

#else

// FFT-16 Adapted from Nussbaumer, "Fast Fourier Transform and Convolution Algorithms"
// 28 FMA + 124 ADD
void fft16(T2 *u) {
  double
      C1 = 0.70710678118654757,   // cos(2t/16)
      C2 = 0.38268343236508978,   // cos(3t/16)
      C3 = 1.3065629648763766,    // cos(3t/16) + cos(t/16)
      C4 = -0.54119610014619701,  // cos(3t/16) - cos(t/16)
      S2 = -0.076120467488713248, // sin(3t/16) - 1
      S3 = 0.54119610014619701,   // sin(3t/16) - sin(t/16)
      S4 = 1.3065629648763766;    // sin(3t/16) + sin(t/16)

  for (int i = 0; i < 8; ++i) { X2(u[i], u[i + 8]); }

  X2(u[0], u[4]);
  X2(u[1], u[5]);
  X2(u[2], u[6]);
  X2(u[3], u[7]);
  X2(u[10], u[14]);
  u[10] = mul_t4(u[10]);

  X2(u[0], u[2]);
  X2(u[1], u[3]);
  X2(u[5], u[7]);
  X2(u[0], u[1]);
  X2(u[9], u[15]);
  X2(u[13], u[11]);

  u[5] = mul_t4(u[5]);
  u[6] = mul_t4(u[6]);
  T2 s3 = fmaT2(C1, u[5],  u[6]);
  T2 s4 = fmaT2(C1, u[5], -u[6]);

  T2 s5 = fmaT2( C1, u[14], u[8]);
  u[5]  = fmaT2(-C1, u[14], u[8]);
  u[14] = s3;
  u[8] = u[1];
  u[1] = s5;

  T2 s1 = fmaT2(C1, u[7], u[4]);
  u[6] = fmaT2(-C1, u[7], u[4]);
  u[4] = u[2];
  u[2] = s1;

  T2 t25 = mul_t4(u[13]);
  T2 m7 = C2 * (u[15] + u[11]);
  T2 s7 = fmaT2(C3, u[15], -m7);
  u[13] = fmaT2(C4, u[11], -m7);

  u[12] = mul_t4(u[12]);
  u[15] = fmaT2( C1, u[10], u[12]);
  u[11] = fmaT2(-C1, u[10], u[12]);
  u[10] = s4;
  u[12] = mul_t4(u[3]);

  T2 t23 = mul_t4(u[9]);
  u[9] = s7;
  u[3] = t23 + t25;
  T2 m15 = fmaT2(S2, u[3], u[3]);
  u[3] = fmaT2(-S4, t25, m15);
  u[7] = fmaT2(-S3, t23, m15);

  X2(u[1], u[9]);
  X2(u[4], u[12]);
  X2(u[5], u[13]);
  X2(u[15], u[7]);
  X2(u[11], u[3]);

  X2(u[1], u[15]);
  X2(u[2], u[14]);
  X2(u[13], u[3]);
  X2(u[5], u[11]);
  X2(u[6], u[10]);
  X2(u[9], u[7]);
}

#endif

// Copyright (C) Mihai Preda & George Woltman

#pragma once

#include "base.cl"

#define A(i) u[(base + i * step) % M]

#if 1
// 10 FMA + 62 ADD
void fft7by(T2 *u, u32 base, u32 step, u32 M) {
  // Adapted from Nussbaumer, "Fast Fourier Transforms and Convolution Algorithms", 5.5.5 7-Point DFT
  const double
      C1=-0.16666666666666666, // (c1 + c2 + c3)/3
      C2=0.79015646852540022,  // (2*c1 - c2 - c3)/3
      C3=0.055854267289647735, // (c1 - 2*c2 + c3)/3
      C4=0.73430220123575241,  // (c1 + c2 - 2*c3)/3
      S1=0.44095855184409843,  // (s1 + s2 - s3)/3
      S2=0.34087293062393137,  // (2*s1 - s2 + s3)/3
      S3=-0.53396936033772513, // (s1 - 2*s2 - s3)/3
      S4=0.87484229096165655;  // (s1 + s2 + 2*s3)/3

  X2(A(1), A(6));
  X2(A(2), A(5));
  X2(A(3), A(4));

  T2 t13 = A(2) - A(1);
  T2 t9  = A(2) - A(3);

  X2(A(1), A(3));

  T2 m2 = -C2 * A(3);
  T2 s0 = fmaT2(C3, t9, m2);
  T2 t4  = A(1) + A(2);
  A(2) = fmaT2(-C4, t13, m2);

  T2 s4 = fmaT2(C1, t4, A(0));
  A(0) = A(0) + t4;
  A(1) = s4 - s0;
  A(3) = s4 + s0 - A(2);
  A(2) = s4      + A(2);

  T2 m6 = -S2 * (A(4) + A(6));
  T2 t2 = fmaT2(S3, A(5) + A(4), m6);
  T2 s3 = fmaT2(S4, A(6) - A(5), m6);

  T2 t1  = S1 * (A(5) - A(4) + A(6));
  A(5) = mul_t4(t1 + s3);
  A(6) = mul_t4(t1 - t2);

  t1 = mul_t4(t1 + t2 - s3);

  X2(A(1), A(6));
  X2(A(2), A(5));

  A(4) = A(3) + t1;
  A(3) = A(3) - t1;

}


#else

// 42 FMA + 18 ADD
void fft7by(T2 *u, u32 base, u32 step, u32 M) {
  // See prime95's gwnum/zr7.mac file for more detailed explanation of the formulas below

  const double
      COS1      =  0.62348980185873348, // cos(tau/7)
      COS2      = -0.22252093395631439, // cos(2*tau/7)
      COS3      = -0.90096886790241915, // cos(3*tau/7)
      SIN1      =  0.7818314824680298,  // sin(tau/7)
      SIN2_SIN1 =  1.246979603717467,   // sin(2*tau/7) / sin(tau/7)
      SIN3_SIN1 =  0.55495813208737121; // sin(3*tau/7) / sin(tau/7)

  X2_mul_t4(A(1), A(6));				// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(A(2), A(5));				// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(A(3), A(4));				// (r4+ i4+),  (i4- -r4-)

  T2 tmp27a = fmaT2(COS1, A(1), A(0));
  T2 tmp36a = fmaT2(COS2, A(1), A(0));
  T2 tmp45a = fmaT2(COS3, A(1), A(0));
  A(0) = A(0) + A(1);

  tmp27a = fmaT2(COS2, A(2), tmp27a);
  tmp36a = fmaT2(COS3, A(2), tmp36a);
  tmp45a = fmaT2(COS1, A(2), tmp45a);
  A(0) = A(0) + A(2);

  tmp27a = fmaT2(COS3, A(3), tmp27a);
  tmp36a = fmaT2(COS1, A(3), tmp36a);
  tmp45a = fmaT2(COS2, A(3), tmp45a);
  A(0) = A(0) + A(3);

  T2 tmp27b = fmaT2(SIN2_SIN1, A(5), A(6));		// .975/.782
  T2 tmp36b = fmaT2(SIN2_SIN1, A(6), -A(4));
  T2 tmp45b = fmaT2(SIN2_SIN1, A(4), -A(5));

  tmp27b = fmaT2(SIN3_SIN1, A(4), tmp27b);		// .434/.782
  tmp36b = fmaT2(SIN3_SIN1, -A(5), tmp36b);
  tmp45b = fmaT2(SIN3_SIN1, A(6), tmp45b);

  fma_addsub(A(1), A(6), SIN1, tmp27a, tmp27b);
  fma_addsub(A(2), A(5), SIN1, tmp36a, tmp36b);
  fma_addsub(A(3), A(4), SIN1, tmp45a, tmp45b);
}
#endif
#undef A

void fft7(T2 *u) { return fft7by(u, 0, 1, 7); }

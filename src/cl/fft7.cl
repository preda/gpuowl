// Copyright (C) Mihai Preda & George Woltman

#pragma once

#include "base.cl"

#if 1
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

#define A(i) u[(base + i * step) % M]

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

#undef A
}

void fft7(T2 *u) { return fft7by(u, 0, 1, 7); }

#else

void fft7(T2 *u, u32 base, u32 step, u32 M) {
  // See prime95's gwnum/zr7.mac file for more detailed explanation of the formulas below
  // R1= r1     +(r2+r7)     +(r3+r6)     +(r4+r5)
  // R2= r1 +.623(r2+r7) -.223(r3+r6) -.901(r4+r5)  +(.782(i2-i7) +.975(i3-i6) +.434(i4-i5))
  // R7= r1 +.623(r2+r7) -.223(r3+r6) -.901(r4+r5)  -(.782(i2-i7) +.975(i3-i6) +.434(i4-i5))
  // R3= r1 -.223(r2+r7) -.901(r3+r6) +.623(r4+r5)  +(.975(i2-i7) -.434(i3-i6) -.782(i4-i5))
  // R6= r1 -.223(r2+r7) -.901(r3+r6) +.623(r4+r5)  -(.975(i2-i7) -.434(i3-i6) -.782(i4-i5))
  // R4= r1 -.901(r2+r7) +.623(r3+r6) -.223(r4+r5)  +(.434(i2-i7) -.782(i3-i6) +.975(i4-i5))
  // R5= r1 -.901(r2+r7) +.623(r3+r6) -.223(r4+r5)  -(.434(i2-i7) -.782(i3-i6) +.975(i4-i5))

  // I1= i1     +(i2+i7)     +(i3+i6)     +(i4+i5)
  // I2= i1 +.623(i2+i7) -.223(i3+i6) -.901(i4+i5)  -(.782(r2-r7) +.975(r3-r6) +.434(r4-r5))
  // I7= i1 +.623(i2+i7) -.223(i3+i6) -.901(i4+i5)  +(.782(r2-r7) +.975(r3-r6) +.434(r4-r5))
  // I3= i1 -.223(i2+i7) -.901(i3+i6) +.623(i4+i5)  -(.975(r2-r7) -.434(r3-r6) -.782(r4-r5))
  // I6= i1 -.223(i2+i7) -.901(i3+i6) +.623(i4+i5)  +(.975(r2-r7) -.434(r3-r6) -.782(r4-r5))
  // I4= i1 -.901(i2+i7) +.623(i3+i6) -.223(i4+i5)  -(.434(r2-r7) -.782(r3-r6) +.975(r4-r5))
  // I5= i1 -.901(i2+i7) +.623(i3+i6) -.223(i4+i5)  +(.434(r2-r7) -.782(r3-r6) +.975(r4-r5))

  const double
      COS1      =  0.62348980185873348, // cos(tau/7)
      COS2      = -0.22252093395631439, // cos(2*tau/7)
      COS3      = -0.90096886790241915, // cos(3*tau/7)
      SIN1      =  0.7818314824680298,  // sin(tau/7)
      SIN2_SIN1 =  1.246979603717467,   // sin(2*tau/7) / sin(tau/7)
      SIN3_SIN1 =  0.55495813208737121; // sin(3*tau/7) / sin(tau/7)

  X2_mul_t4(u[1], u[6]);				// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[5]);				// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(u[3], u[4]);				// (r4+ i4+),  (i4- -r4-)

  T2 tmp27a = fmaT2(COS1, u[1], u[0]);
  T2 tmp36a = fmaT2(COS2, u[1], u[0]);
  T2 tmp45a = fmaT2(COS3, u[1], u[0]);
  u[0] = u[0] + u[1];

  tmp27a = fmaT2(COS2, u[2], tmp27a);
  tmp36a = fmaT2(COS3, u[2], tmp36a);
  tmp45a = fmaT2(COS1, u[2], tmp45a);
  u[0] = u[0] + u[2];

  tmp27a = fmaT2(COS3, u[3], tmp27a);
  tmp36a = fmaT2(COS1, u[3], tmp36a);
  tmp45a = fmaT2(COS2, u[3], tmp45a);
  u[0] = u[0] + u[3];

  T2 tmp27b = fmaT2(SIN2_SIN1, u[5], u[6]);		// .975/.782
  T2 tmp36b = fmaT2(SIN2_SIN1, u[6], -u[4]);
  T2 tmp45b = fmaT2(SIN2_SIN1, u[4], -u[5]);

  tmp27b = fmaT2(SIN3_SIN1, u[4], tmp27b);		// .434/.782
  tmp36b = fmaT2(SIN3_SIN1, -u[5], tmp36b);
  tmp45b = fmaT2(SIN3_SIN1, u[6], tmp45b);

  fma_addsub(u[1], u[6], SIN1, tmp27a, tmp27b);
  fma_addsub(u[2], u[5], SIN1, tmp36a, tmp36b);
  fma_addsub(u[3], u[4], SIN1, tmp45a, tmp45b);
}
#endif

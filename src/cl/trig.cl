// Copyright (C) George Woltman and Mihai Preda

#pragma once

// Copyright notice of original k_cos, k_sin from which our ksin/kcos evolved:
/* ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

// Coefficients from http://www.netlib.org/fdlibm/k_cos.c
#define COS_COEFS {-0.5,0.041666666666666602,-0.001388888888887411,2.4801587289476729e-05,-2.7557314351390663e-07,2.0875723212981748e-09,-1.1359647557788195e-11, M_PI}

// Experimental: const double C[] = {-0.5,0.041666666666665589,-0.0013888888888779014,2.4801587246942509e-05,-2.7557304501248813e-07,2.0874583610048953e-09,-1.1307548621486489e-11, M_PI};

double ksinpi(u32 k, u32 n) {
  // const double S[] = {-0.16666666666666455,0.0083333333332988729,-0.00019841269816529426,2.7557310051600518e-06,-2.5050279451232251e-08,1.5872611854244144e-10};

  // Coefficients from http://www.netlib.org/fdlibm/k_sin.c
  const double S[] = {-0.16666666666666666,0.0083333333333309497,-0.00019841269836761127,2.7557316103728802e-06,-2.5051132068021698e-08,1.5918144304485914e-10, M_PI};

  double x = S[6] / n * k;
  double z = x * x;
  double r = fma(fma(fma(fma(fma(S[5], z, S[4]), z, S[3]), z, S[2]), z, S[1]), z, S[0]);

  // return fma(r * z, x, x);
  // return fma(r, z, 1) * x;
  // return fma(r * x, z, x);
  return fma(r, z * x, x);
}

double kcospi(u32 k, u32 n) {
  const double C[] = COS_COEFS;
  double x = C[7] / n * k;
  double z = x * x;
  return fma(fma(fma(fma(fma(fma(fma(C[6], z, C[5]), z, C[4]), z, C[3]), z, C[2]), z, C[1]), z, C[0]), z, 1);
}

// N represents a full circle, so N/2 is pi radians and N/8 is pi/4 radians.
double2 reducedCosSin(u32 k, u32 N) {
  assert(k <= N/8);
  return U2(kcospi(k, N/2), -ksinpi(k, N/2));
}

// Returns e^(-i * tau * k / n), (tau == 2*pi represents a full circle). So k/n is the ratio of a full circle.
// Inverse trigonometric direction is chosen as an FFT convention.
double2 slowTrig_N(u32 k, u32 kBound)   {
  u32 n = ND;
  assert(n % 8 == 0);
  assert(k < kBound);       // kBound actually bounds k
  assert(kBound <= 2 * n);  // angle <= 2 tau

  if (kBound > n && k >= n) { k -= n; }
  assert(k < n);

  bool negate = kBound > n/2 && k >= n/2;
  if (negate) { k -= n/2; }

  bool negateCos = kBound > n / 4 && k >= n / 4;
  if (negateCos) { k = n/2 - k; }

  bool flip = kBound > n / 8 + 1 && k > n / 8;
  if (flip) { k = n / 4 - k; }

  assert(k <= n / 8);

  double2 r = reducedCosSin(k, n);

  if (flip) { r = -swap(r); }
  if (negateCos) { r.x = -r.x; }
  if (negate) { r = -r; }

  return r;
}

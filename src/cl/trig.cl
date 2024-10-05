// Copyright (C) George Woltman and Mihai Preda

#pragma once

#include "math.cl"

// N represents a full circle, so N/2 is pi radians and N/8 is pi/4 radians.
double2 reducedCosSin(int k, u32 N) {
  assert(k >= -N/8 && k <= N/8);

  const double S[] = TRIG_SIN;
  double x = k * -TRIG_SMUL;
  double z = x * x;

  double r = S[7];
  r = fma(r, z, S[6]);
  r = fma(r, z, S[5]);
  r = fma(r, z, S[4]);
  r = fma(r, z, S[3]);
  r = fma(r, z, S[2]);
  r = fma(r, z, S[0]);
  double s = fma(x, S[1], r * x);

  const double C[] = TRIG_COS;
  x = k * TRIG_CMUL;
  z = x * x;

  r = C[7];
  r = fma(r, z, C[6]);
  r = fma(r, z, C[5]);
  r = fma(r, z, C[4]);
  r = fma(r, z, C[3]);
  r = fma(r, z, C[2]);
  r = fma(r, z, C[1]);
  double c = fma(r, z, 1); // C[0] == 1

  return U2(c, s);
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

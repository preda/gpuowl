// Copyright (C) George Woltman and Mihai Preda

#pragma once

#include "math.cl"

double2 reducedCosSin(int k, double cosBase) {
  const double S[] = TRIG_SIN;
  const double C[] = TRIG_COS;

  double x = k * TRIG_SCALE;
  double z = x * x;

  double r1 = fma(S[7], z, S[6]);
  double r2 = fma(C[7], z, C[6]);

  r1 = fma(r1, z, S[5]);
  r2 = fma(r2, z, C[5]);

  r1 = fma(r1, z, S[4]);
  r2 = fma(r2, z, C[4]);

  r1 = fma(r1, z, S[3]);
  r2 = fma(r2, z, C[3]);

  r1 = fma(r1, z, S[2]);
  r2 = fma(r2, z, C[2]);

  r1 = fma(r1, z, S[1]);
  r2 = fma(r2, z, C[1]);

  r1 = r1 * x;
  double c = fma(r2, z, cosBase);
  double s = fma(x, S[0], r1);

  return U2(c, s);
}

double2 fancyTrig_N(u32 k) {
  return reducedCosSin(k, 0);
}

// Returns e^(i * tau * k / n), (tau == 2*pi represents a full circle). So k/n is the ratio of a full circle.
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

  double2 r = reducedCosSin(k, 1);

  if (flip) { r = swap(r); }
  if (negateCos) { r.x = -r.x; }
  if (negate) { r = -r; }

  return r;
}

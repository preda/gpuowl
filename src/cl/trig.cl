// Copyright (C) George Woltman and Mihai Preda

#pragma once

#include "math.cl"

double2 reducedCosSin(int k) {
  const double S[] = TRIG_SIN;
  const double C[] = TRIG_COS;

  double x = k * -TRIG_SCALE;
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
  double c = fma(r2, z, C[0]); // C[0] == 1
  double s = fma(x, S[0], r1);

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

  double2 r = reducedCosSin(k);

  if (flip) { r = -swap(r); }
  if (negateCos) { r.x = -r.x; }
  if (negate) { r = -r; }

  return r;
}

int div15(int k) {
  // assert(k >= 0 && k < 150
  return k >= 75
            ? k >= 105 ? (k >= 135 ? 9 : (k >= 120 ? 8 : 7)) : (k >= 90 ? 6 : 5)
            : (k >= 45 ? (k >= 60 ? 4 : 3) : (k >= 30 ? 2 : k >= 15 ? 1 : 0));
}

double2 reducedCosSin_128(int k) {
  double TAB[8*3 + 10*3] = {
  0,0,0,
  -1.8824717398857331e-05,-0.0061358846491544735,-1.825303094274199e-18,
  -7.5298160855459049e-05,-0.012271538285719924,-2.4268852923787619e-18,
  -0.00016941820417657797,-0.01840672990580482,-7.4208106016889887e-19,
  -0.00030118130379577979,-0.024541228522912285,-3.3785960317856879e-18,
  -0.00047058249890683673,-0.030674803176636619,-6.9252167348596822e-18,
  -0.00067761541165049907,-0.036807222941358832,-6.1101492116262661e-19,
  -0.00092227224735461728,-0.042938256934940827,4.107488067291941e-18,

  0,0,0,
  -0.004232585532340208,-0.091908956497132752,2.309014715298077e-17,
  -0.016894512568783666,-0.18303988795514092,-3.6100467282739861e-17,
  -0.037878595730958434,-0.27262135544994909,1.0721368987029825e-16,
  -0.06700720116526114,-0.35989503653498822,7.8363775633007422e-17,
  -0.10403375024381491,-0.44412214457042937,1.5094449921690295e-16,
  -0.14864480689473475,-0.52458968267846873,-2.1022451890161859e-16,
  -0.20046273089209518,-0.6006164793838692,3.3574770498692591e-16,
  -0.25904887464504095,-0.67155895484701844,5.4644680821424042e-17,
  -0.32390729642468419,-0.73681656887737002,2.0700529649863492e-16,
  };

  int bin = (k + 7) / 15;

  int p = k - 15 * bin;

  int p3 = abs(p) * 3;
  double c1 = TAB[p3];
  double s1 = TAB[p3 + 1];
  double d1 = TAB[p3 + 2];
  if (p < 0) {
    s1 = -s1;
    d1 = -d1;
  }

  p3 = 8*3 + bin * 3;
  double c2 = TAB[p3];
  double s2 = TAB[p3 + 1];
  double d2 = TAB[p3 + 2];
  double d = d1 + d2;

  c1 = fma(-s1, d, c1);
  s1 += d;

  double c  = fma(-s1, s2, c2) + fma(c1, c2, c1) + 1;
  double s  = fma(c2, s1, fma(s2, c1, s1)) + s2;
  return U2(c, s);
}

double2 trig_1024(u32 k, u32 kBound)   {
  u32 n = 1024;
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

  double2 r = reducedCosSin_128(k);

  if (flip) { r = -swap(r); }
  if (negateCos) { r.x = -r.x; }
  if (negate) { r = -r; }

  return r;
}


// Copyright (C) George Woltman and Mihai Preda

#pragma once

#if ULTRA_TRIG

// These are ultra accurate routines.  We modified Ernst's qfcheb program and selected a multiplier such that
// a) k * multipler / n can be represented exactly as a double, and
// b) x * x can be represented exactly as a double, and
// c) the difference between S0 and C0 represented as a double vs infinite precision is minimized.
// Note that condition (a) requires different multipliers for different MIDDLE values.

#if MIDDLE <= 4 || MIDDLE == 6 || MIDDLE == 8 || MIDDLE == 12

#define SIN_COEFS {0.013255665205020225,-3.8819803226819742e-07,3.4105654433606424e-12,-1.4268560139781677e-17,3.4821751757020666e-23,-5.5620764489252689e-29,6.2011635226098908e-35, 237}
#define COS_COEFS {-8.7856330013791936e-05,1.2864557872487131e-09,-7.5348856128299892e-15,2.3642407019488875e-20,-4.6158547847666762e-26,6.1440808274170587e-32,-5.8714657758002626e-38, 237}

#elif MIDDLE == 11

#define SIN_COEFS {0.0058285577988678909,-3.3001377814174114e-08,5.6056282285321817e-14,-4.5341639101320423e-20,2.1393746357239815e-26,-6.6068179928427645e-33,1.4241237670800455e-39, 49*11}

// {0.005492294848933205,-2.761278944626038e-08,4.1647407612374865e-14,-2.9912063263788177e-20,1.2532031863362935e-26,-3.4364949357849832e-33,6.5816772637974481e-40, 143 * 4};

#define COS_COEFS {-1.6986043007371857e-05,4.8087609508047477e-11,-5.4454546881584527e-17,3.3034545522108849e-23,-1.2469468591680302e-29,3.2090160573145604e-36,-5.9289892824449386e-43, 49*11}

// {-1.5082651353809108e-05,3.7914395310092582e-11,-3.8123307049954028e-17,2.0535733847563737e-23,-6.8829596395036851e-30,1.5727926864212422e-36,-2.5737325744028274e-43, 143 * 4};

// This should be the best choice for MIDDLE=11.  For reasons I cannot explain, the Sun coefficients beat this
// code.  We know this code works as it gives great results for MIDDLE=5 and MIDDLEE=10.
#elif MIDDLE == 5 || MIDDLE == 10 || MIDDLE == 11

#define SIN_COEFS {0.0033599921428767842,-6.3221316482145663e-09,3.5687001824123009e-15,-9.5926212207432193e-22,1.5041156546369205e-28,-1.5436222869257443e-35,1.1057341951605355e-42, 85 * 11}
#define COS_COEFS {-5.6447736000968621e-06,5.3105781660583875e-12,-1.9984674288638241e-18,4.0288914910974918e-25,-5.0538167304629085e-32,4.3221291704923216e-39,-2.6537550015407366e-46, 85 * 11}

#elif MIDDLE == 7 || MIDDLE == 14

#define SIN_COEFS {0.0030120734933746819,-4.5545496673734544e-09,2.066077343547647e-15,-4.4630156850662332e-22,5.6237622654854882e-29,-4.6381134477150518e-36,2.6699656391050201e-43, 149 * 7}
#define COS_COEFS {-4.5362933647451799e-06,3.4296595818385068e-12,-1.0371961336265129e-18,1.6803664055570525e-25,-1.6939185081650983e-32,1.1641940392856976e-39,-5.7443786570712859e-47, 149 * 7}

#elif MIDDLE == 9

#define SIN_COEFS {0.0032024389944850084,-5.4738305054252556e-09,2.8068750524532041e-15,-6.8538645985346134e-22,9.7625812823195236e-29,-9.1014309535685973e-36,5.9224934064240749e-43, 109*9}
#define COS_COEFS {-5.1278077566990758e-06,4.3824020649438457e-12,-1.4981410201032161e-18,2.7436354064447543e-25,-3.1264070669243379e-32,2.4288963593034543e-39,-1.3547441195887408e-46,109*9}

#elif MIDDLE == 13

#define SIN_COEFS {0.0034036756810290284,-6.5719350793639721e-09,3.8067960700283384e-15,-1.0500419865908618e-21,1.6895475541832181e-28,-1.779303563885441e-35,1.3079151443222568e-42, 71*13}
#define COS_COEFS {-5.7925040708142102e-06,5.5921839017331711e-12,-2.1595165343644285e-18,4.4675029669254463e-25,-5.7506718639750315e-32,5.0468065746580596e-39,-3.1797982320621979e-46, 71*13}

#elif MIDDLE == 15

#define SIN_COEFS {0.0032221463113741469,-5.5755089924509359e-09,2.8943099587177684e-15,-7.1546148933480147e-22,1.0316780436916074e-28,-9.7368389615915834e-36,6.4141878934890016e-43, 65*15}
#define COS_COEFS {-5.1911134259510105e-06,4.4912764335147829e-12,-1.5543150262419615e-18,2.8816519982635366e-25,-3.3242175693980929e-32,2.6144580878684214e-39,-1.4762460824550342e-46, 65*15}

#endif

double ksinpi(u32 k, u32 n) {
  const double S[] = SIN_COEFS;

  double x = S[7] / n * k;
  double z = x * x;
  double r = fma(fma(fma(fma(fma(S[6], z, S[5]), z, S[4]), z, S[3]), z, S[2]), z, S[1]) * (z * x);
  return fma(x, S[0], r);
}

#else

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
  const double S[] = {1, -0.16666666666666666,0.0083333333333309497,-0.00019841269836761127,2.7557316103728802e-06,-2.5051132068021698e-08,1.5918144304485914e-10, M_PI};

  double x = S[7] / n * k;
  double z = x * x;
  // Special-case based on S[0]==1:
  return fma(fma(fma(fma(fma(fma(S[6], z, S[5]), z, S[4]), z, S[3]), z, S[2]), z, S[1]), z * x, x);
}

#endif

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

#if !defined(TRIG_TAB)
#define TRIG_TAB 0
#endif

// Returns e^(-i * tau * k / n), (tau == 2*pi represents a full circle). So k/n is the ratio of a full circle.
// Inverse trigonometric direction is chosen as an FFT convention.
double2 slowTrig_N(u32 k, u32 kBound, BigTab TRIG_BHW)   {
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

  double2 r;

  if (TRIG_TAB & (TRIG_BHW != NULL)) { // bitwise because annoying warning -Wconstant-logical-operand
    u32 a = (k + WIDTH/2) / WIDTH;
    i32 b = k - a * WIDTH;

    double2 cs1 = TRIG_BHW[a];
    if (b == 0) {
      r = cs1;
    } else {
      double2 cs2 = TRIG_BHW[BIG_HEIGHT/8 + 1 + abs(b)];
      r = cmulFancy(cs1, b < 0 ? conjugate(cs2) : cs2);
    }
  } else {
    r = reducedCosSin(k, n);
  }

  if (flip) { r = -swap(r); }
  if (negateCos) { r.x = -r.x; }
  if (negate) { r = -r; }

  return r;
}

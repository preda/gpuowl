// Copyright (C) Mihai Preda and George Woltman

void fft14(T2 *u) {
  const double
      COS1 = 0.62348980185873348,
      COS2 = -0.22252093395631439,
      COS3 = -0.90096886790241915,
      SIN1 = 0.7818314824680298,
      SIN2_SIN1 = 1.246979603717467,
      SIN3_SIN1 = 0.55495813208737121;

  /*
  const double SIN1 = 0.781831482468029809;		// sin(tau/7)
  const double SIN2_SIN1 = 1.2469796037174670611;	// sin(2*tau/7) / sin(tau/7) = .975/.782
  const double SIN3_SIN1 = 0.5549581320873711914;	// sin(3*tau/7) / sin(tau/7) = .434/.782
  const double COS1 = 0.6234898018587335305;		// cos(tau/7)
  const double COS2 = -0.2225209339563144043;		// cos(2*tau/7)
  const double COS3 = -0.9009688679024191262;		// cos(3*tau/7)
  */

  X2(u[0], u[7]);					// (r1+ i1+),  (r1-  i1-)
  X2_mul_t4(u[1], u[8]);				// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[9]);				// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(u[3], u[10]);				// (r4+ i4+),  (i4- -r4-)
  X2_mul_t4(u[4], u[11]);				// (r5+ i5+),  (i5- -r5-)
  X2_mul_t4(u[5], u[12]);				// (r6+ i6+),  (i6- -r6-)
  X2_mul_t4(u[6], u[13]);				// (r7+ i7+),  (i7- -r7-)

  X2_mul_t4(u[1], u[6]);				// (r2++  i2++),  (i2+- -r2+-)
  X2_mul_t4(u[2], u[5]);				// (r3++  i3++),  (i3+- -r3+-)
  X2_mul_t4(u[3], u[4]);				// (r4++  i4++),  (i4+- -r4+-)
  X2_mul_t4(u[8], u[13]);				// (i2-+ -r2-+), (-r2-- -i2--)
  X2_mul_t4(u[9], u[12]);				// (i3-+ -r3-+), (-r3-- -i3--)
  X2_mul_t4(u[10], u[11]);				// (i4-+ -r4-+), (-r4-- -i4--)

  T2 tmp313a = fmaT2(COS1, u[1], u[0]);
  T2 tmp511a = fmaT2(COS2, u[1], u[0]);
  T2 tmp79a = fmaT2(COS3, u[1], u[0]);
  u[0] = u[0] + u[1];
  T2 tmp214a = fmaT2(COS3, u[13], u[7]);
  T2 tmp412a = fmaT2(COS2, u[13], u[7]);
  T2 tmp610a = fmaT2(COS1, u[13], u[7]);
  u[7] = u[7] + u[13];

  tmp313a = fmaT2(COS2, u[2], tmp313a);
  tmp511a = fmaT2(COS3, u[2], tmp511a);
  tmp79a = fmaT2(COS1, u[2], tmp79a);
  u[0] = u[0] + u[2];
  tmp214a = fmaT2(COS1, -u[12], tmp214a);
  tmp412a = fmaT2(COS3, -u[12], tmp412a);
  tmp610a = fmaT2(COS2, -u[12], tmp610a);
  u[7] = u[7] - u[12];

  tmp313a = fmaT2(COS3, u[3], tmp313a);
  tmp511a = fmaT2(COS1, u[3], tmp511a);
  tmp79a = fmaT2(COS2, u[3], tmp79a);
  u[0] = u[0] + u[3];
  tmp214a = fmaT2(COS2, u[11], tmp214a);
  tmp412a = fmaT2(COS1, u[11], tmp412a);
  tmp610a = fmaT2(COS3, u[11], tmp610a);
  u[7] = u[7] + u[11];

  T2 tmp313b = fmaT2(SIN2_SIN1, u[5], u[6]);			// Apply .975/.782
  T2 tmp511b = fmaT2(SIN2_SIN1, u[6], -u[4]);
  T2 tmp79b = fmaT2(SIN2_SIN1, u[4], -u[5]);
  T2 tmp214b = fmaT2(SIN2_SIN1, u[10], u[9]);
  T2 tmp412b = fmaT2(SIN2_SIN1, u[8], -u[10]);
  T2 tmp610b = fmaT2(SIN2_SIN1, -u[9], u[8]);

  tmp313b = fmaT2(SIN3_SIN1, u[4], tmp313b);			// Apply .434/.782
  tmp511b = fmaT2(SIN3_SIN1, -u[5], tmp511b);
  tmp79b = fmaT2(SIN3_SIN1, u[6], tmp79b);
  tmp214b = fmaT2(SIN3_SIN1, u[8], tmp214b);
  tmp412b = fmaT2(SIN3_SIN1, u[9], tmp412b);
  tmp610b = fmaT2(SIN3_SIN1, u[10], tmp610b);

  fma_addsub(u[1], u[13], SIN1, tmp214a, tmp214b);
  fma_addsub(u[2], u[12], SIN1, tmp313a, tmp313b);
  fma_addsub(u[3], u[11], SIN1, tmp412a, tmp412b);
  fma_addsub(u[4], u[10], SIN1, tmp511a, tmp511b);
  fma_addsub(u[5], u[9], SIN1, tmp610a, tmp610b);
  fma_addsub(u[6], u[8], SIN1, tmp79a, tmp79b);
}

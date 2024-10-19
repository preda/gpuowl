// Copyright (C) Mihai Preda & George Woltman

#if 0
// Adapted from https://web.archive.org/web/20101126231320/http://cnx.org/content/col10569/1.7/pdf
// 40 FMA + 150 ADD
void fft11(T2 *u) {
  double
      C1 = 0.1, C2 = 0.33166247903553997,
      C3 = 0.51541501300188641, C4 = 0.94125353283118118,
      C5 = 1.4143537075597825, C6 = 0.85949297361449739,
      C7 = 0.042314838273285138, C8 = 0.38639279888589612,
      C9 = 0.51254589567199993, C10 = 1.0702757469471715,
      C11 = 0.55486073394528501, C12 = 1.2412944743900585,
      C13 = 0.20897833842005759, C14 = 0.374157173124608,
      C15 = 0.049929922194110285, C16 = 0.65815896284539277,
      C17 = 0.63306543373877588, C18 = 1.0822460581641111,
      C19 = 0.81720737907134011, C20 = 0.4240870953187183;

  X2(u[1], u[10]);
  X2(u[2], u[9]);
  X2(u[3], u[8]);
  X2(u[4], u[7]);
  X2(u[5], u[6]);

  T2 t1 = u[1];
  T2 t6 = u[10];

  T2 t2 = u[2];
  T2 t7 = u[9];

  T2 t3 = u[3];
  T2 t8 = u[8];

  T2 t4 = u[4];
  T2 t9 = u[7];

  T2 t5  = u[5];
  T2 t10 = u[6];

  T2 t11 = t1 + t2;
  T2 t12 = t3 + t5;
  T2 t13 = t4 + t11 + t12;
  T2 s0 = u[0] - C1 * t13;
  u[0]  = u[0] + t13;

  T2 t14 = t7 - t8;
  T2 t15 = t6 + t10;

  T2 am2 = (t14 - t15 - t9) * C2;
  T2 am3 = (t2 - t4) * C3;
  T2 am4 = (t1 - t4) * C4;
  T2 am5 = (t2 - t1) * C5;
  T2 am6 = (t5 - t4) * C6;
  T2 am7 = (t3 - t4) * C7;
  T2 am8 = (t5 - t3) * C8;
  T2 am11 = (t12 - t11) * C11;
  T2 am14 = (t6 + t7) * C14;
  T2 am17 = (t8 - t10) * C17;
  T2 am20 = (t14 + t15) * C20;

  T2 s7 = am11 + C10 * (t1 - t3);
  T2 s8 = am11 + (t2 - t5) * C9;
  T2 s9 = am14 + (t6 - t9) * C13;
  T2 s10 = -am14 + (t7 + t9) * C12;
  T2 s11 = am17 + (t8 - t9) * C16;
  T2 s12 = -am17 + (t9 - t10) * C15;
  T2 s13 = am20 + (t6 - t8) * C19;
  T2 s14 = -am20 + (t7 + t10) * C18;

  u[1] = s0 + am3 + am4 - am6 - am7;
  u[10] = mul_t4(s9 + s10 + s11 + s12 - am2);
  X2(u[1], u[10]);

  u[2] = s0 + s7 + am7 + am8;
  u[9] = mul_t4(s13 + am2 + s11);
  X2(u[2], u[9]);

  u[3] = s0 - s7 - am4 - am5;
  u[8] = mul_t4(s13 - am2 - s9);
  X2(u[3], u[8]);

  u[4] = s0 + s8 + am6 - am8;
  u[7] = -mul_t4(s14 + am2 + s12);
  X2(u[4], u[7]);

  u[5] = s0 - s8 - am3 + am5;
  u[6] = mul_t4(s14 - am2 - s10);
  X2(u[5], u[6]);
}

#else

// 110 FMA + 30 ADD (or 100 FMA + 50 ADD)
// See prime95's gwnum/zr11.mac file for more detailed explanation

void fft11(T2 *u) {
  const double COS1 = 0.8412535328311811688;		// cos(tau/11)
  const double COS2 = 0.4154150130018864255;		// cos(2*tau/11)
  const double COS3 = -0.1423148382732851404;		// cos(3*tau/11)
  const double COS4 = -0.6548607339452850640;		// cos(4*tau/11)
  const double COS5 = -0.9594929736144973898;		// cos(5*tau/11)
  const double SIN1 = 0.5406408174555975821;		// sin(tau/11)
  const double SIN2_SIN1 = 1.682507065662362337;	// sin(2*tau/11) / sin(tau/11) = .910/.541
  const double SIN3_SIN1 = 1.830830026003772851;	// sin(3*tau/11) / sin(tau/11) = .990/.541
  const double SIN4_SIN1 = 1.397877389115792056;	// sin(4*tau/11) / sin(tau/11) = .756/.541
  const double SIN5_SIN1 = 0.521108558113202723;	// sin(5*tau/11) / sin(tau/11) = .282/.541

  X2_mul_t4(u[1], u[10]);				// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[9]);				// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(u[3], u[8]);				// (r4+ i4+),  (i4- -r4-)
  X2_mul_t4(u[4], u[7]);				// (r5+ i5+),  (i5- -r5-)
  X2_mul_t4(u[5], u[6]);				// (r6+ i6+),  (i6- -r6-)

  T2 tmp211a = fmaT2(COS1, u[1], u[0]);
  T2 tmp310a = fmaT2(COS2, u[1], u[0]);
  T2 tmp49a = fmaT2(COS3, u[1], u[0]);
  T2 tmp58a = fmaT2(COS4, u[1], u[0]);
  T2 tmp67a = fmaT2(COS5, u[1], u[0]);
  u[0] = u[0] + u[1];

  tmp211a = fmaT2(COS2, u[2], tmp211a);
  tmp310a = fmaT2(COS4, u[2], tmp310a);
  tmp49a = fmaT2(COS5, u[2], tmp49a);
  tmp58a = fmaT2(COS3, u[2], tmp58a);
  tmp67a = fmaT2(COS1, u[2], tmp67a);
  u[0] = u[0] + u[2];

  tmp211a = fmaT2(COS3, u[3], tmp211a);
  tmp310a = fmaT2(COS5, u[3], tmp310a);
  tmp49a = fmaT2(COS2, u[3], tmp49a);
  tmp58a = fmaT2(COS1, u[3], tmp58a);
  tmp67a = fmaT2(COS4, u[3], tmp67a);
  u[0] = u[0] + u[3];

  tmp211a = fmaT2(COS4, u[4], tmp211a);
  tmp310a = fmaT2(COS3, u[4], tmp310a);
  tmp49a = fmaT2(COS1, u[4], tmp49a);
  tmp58a = fmaT2(COS5, u[4], tmp58a);
  tmp67a = fmaT2(COS2, u[4], tmp67a);
  u[0] = u[0] + u[4];

  tmp211a = fmaT2(COS5, u[5], tmp211a);
  tmp310a = fmaT2(COS1, u[5], tmp310a);
  tmp49a = fmaT2(COS4, u[5], tmp49a);
  tmp58a = fmaT2(COS2, u[5], tmp58a);
  tmp67a = fmaT2(COS3, u[5], tmp67a);
  u[0] = u[0] + u[5];

  T2 tmp211b = fmaT2(SIN2_SIN1, u[9], u[10]);		// .910/.541
  T2 tmp310b = fmaT2(SIN2_SIN1, u[10], -u[6]);
  T2 tmp49b = fmaT2(SIN2_SIN1, -u[8], u[7]);
  T2 tmp58b = fmaT2(SIN2_SIN1, -u[6], u[8]);
  T2 tmp67b = fmaT2(SIN2_SIN1, -u[7], -u[9]);

  tmp211b = fmaT2(SIN3_SIN1, u[8], tmp211b);		// .990/.541
  tmp310b = fmaT2(SIN3_SIN1, -u[7], tmp310b);
  tmp49b = fmaT2(SIN3_SIN1, u[10], tmp49b);
  tmp58b = fmaT2(SIN3_SIN1, -u[9], tmp58b);
  tmp67b = fmaT2(SIN3_SIN1, u[6], tmp67b);

  tmp211b = fmaT2(SIN4_SIN1, u[7], tmp211b);		// .756/.541
  tmp310b = fmaT2(SIN4_SIN1, u[9], tmp310b);
  tmp49b = fmaT2(SIN4_SIN1, u[6], tmp49b);
  tmp58b = fmaT2(SIN4_SIN1, u[10], tmp58b);
  tmp67b = fmaT2(SIN4_SIN1, u[8], tmp67b);

  tmp211b = fmaT2(SIN5_SIN1, u[6], tmp211b);		// .282/.541
  tmp310b = fmaT2(SIN5_SIN1, -u[8], tmp310b);
  tmp49b = fmaT2(SIN5_SIN1, -u[9], tmp49b);
  tmp58b = fmaT2(SIN5_SIN1, u[7], tmp58b);
  tmp67b = fmaT2(SIN5_SIN1, u[10], tmp67b);

  fma_addsub(u[1], u[10], SIN1, tmp211a, tmp211b);
  fma_addsub(u[2], u[9], SIN1, tmp310a, tmp310b);
  fma_addsub(u[3], u[8], SIN1, tmp49a, tmp49b);
  fma_addsub(u[4], u[7], SIN1, tmp58a, tmp58b);
  fma_addsub(u[5], u[6], SIN1, tmp67a, tmp67b);
}

#endif

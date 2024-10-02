// Copyright (C) Mihai Preda and George Woltman

// To calculate a 13-complex FFT in a brute force way (using a shorthand notation):
// The sin/cos values (w = 13th root of unity) are:
// w^1 = .885 - .465i
// w^2 = .568 - .823i
// w^3 = .121 - .993i
// w^4 = -.355 - .935i
// w^5 = -.749 - .663i
// w^6 = -.971 - .239i
// w^7 = -.971 + .239i
// w^8 = -.749 + .663i
// w^9 = -.355 + .935i
// w^10= .121 + .993i
// w^11= .568 + .823i
// w^12= .885 + .465i
//
// R1 = r1     +(r2+r13)     +(r3+r12)     +(r4+r11)     +(r5+r10)     +(r6+r9)     +(r7+r8)
// R2 = r1 +.885(r2+r13) +.568(r3+r12) +.121(r4+r11) -.355(r5+r10) -.749(r6+r9) -.971(r7+r8)  +.465(i2-i13) +.823(i3-i12) +.993(i4-i11) +.935(i5-i10) +.663(i6-i9) +.239(i7-i8)
// R13= r1 +.885(r2+r13) +.568(r3+r12) +.121(r4+r11) -.355(r5+r10) -.749(r6+r9) -.971(r7+r8)  -.465(i2-i13) -.823(i3-i12) -.993(i4-i11) -.935(i5-i10) -.663(i6-i9) -.239(i7-i8)
// R3 = r1 +.568(r2+r13) -.355(r3+r12) -.971(r4+r11) -.749(r5+r10) +.121(r6+r9) +.885(r7+r8)  +.823(i2-i13) +.935(i3-i12) +.239(i4-i11) -.663(i5-i10) -.993(i6-i9) -.465(i7-i8)
// R12= r1 +.568(r2+r13) -.355(r3+r12) -.971(r4+r11) -.749(r5+r10) +.121(r6+r9) +.885(r7+r8)  -.823(i2-i13) -.935(i3-i12) -.239(i4-i11) +.663(i5-i10) +.993(i6-i9) +.465(i7-i8)
// R4 = r1 +.121(r2+r13) -.971(r3+r12) -.355(r4+r11) +.885(r5+r10) +.568(r6+r9) -.749(r7+r8)  +.993(i2-i13) +.239(i3-i12) -.935(i4-i11) -.465(i5-i10) +.823(i6-i9) +.663(i7-i8)
// R11= r1 +.121(r2+r13) -.971(r3+r12) -.355(r4+r11) +.885(r5+r10) +.568(r6+r9) -.749(r7+r8)  -.993(i2-i13) -.239(i3-i12) +.935(i4-i11) +.465(i5-i10) -.823(i6-i9) -.663(i7-i8)
// R5 = r1 -.355(r2+r13) -.749(r3+r12) +.885(r4+r11) +.121(r5+r10) -.971(r6+r9) +.568(r7+r8)  +.935(i2-i13) -.663(i3-i12) -.465(i4-i11) +.993(i5-i10) -.239(i6-i9) -.823(i7-i8)
// R10= r1 -.355(r2+r13) -.749(r3+r12) +.885(r4+r11) +.121(r5+r10) -.971(r6+r9) +.568(r7+r8)  -.935(i2-i13) +.663(i3-i12) +.465(i4-i11) -.993(i5-i10) +.239(i6-i9) +.823(i7-i8)
// R6 = r1 -.749(r2+r13) +.121(r3+r12) +.568(r4+r11) -.971(r5+r10) +.885(r6+r9) -.355(r7+r8)  +.663(i2-i13) -.993(i3-i12) +.823(i4-i11) -.239(i5-i10) -.465(i6-i9) +.935(i7-i8)
// R9 = r1 -.749(r2+r13) +.121(r3+r12) +.568(r4+r11) -.971(r5+r10) +.885(r6+r9) -.355(r7+r8)  -.663(i2-i13) +.993(i3-i12) -.823(i4-i11) +.239(i5-i10) +.465(i6-i9) -.935(i7-i8)
// R7 = r1 -.971(r2+r13) +.885(r3+r12) -.749(r4+r11) +.568(r5+r10) -.355(r6+r9) +.121(r7+r8)  +.239(i2-i13) -.465(i3-i12) +.663(i4-i11) -.823(i5-i10) +.935(i6-i9) -.993(i7-i8)
// R8 = r1 -.971(r2+r13) +.885(r3+r12) -.749(r4+r11) +.568(r5+r10) -.355(r6+r9) +.121(r7+r8)  -.239(i2-i13) +.465(i3-i12) -.663(i4-i11) +.823(i5-i10) -.935(i6-i9) +.993(i7-i8)
//
// I1 = i1                                                                                        +(i2+i13)     +(i3+i12)     +(i4+i11)     +(i5+i10)     +(i6+i9)     +(i7+i8)
// I2 = i1 -.465(r2-r13) -.823(r3-r12) -.993(r4-r11) -.935(r5-r10) -.663(r6-r9) -.239(r7-r8)  +.885(i2+i13) +.568(i3+i12) +.121(i4+i11) -.355(i5+i10) -.749(i6+i9) -.971(i7+i8)
// I13= i1 +.465(r2-r13) +.823(r3-r12) +.993(r4-r11) +.935(r5-r10) +.663(r6-r9) +.239(r7-r8)  +.885(i2+i13) +.568(i3+i12) +.121(i4+i11) -.355(i5+i10) -.749(i6+i9) -.971(i7+i8)
// I3 = i1 -.823(r2-r13) -.935(r3-r12) -.239(r4-r11) +.663(r5-r10) +.993(r6-r9) +.465(r7-r8)  +.568(i2+i13) -.355(i3+i12) -.971(i4+i11) -.749(i5+i10) +.121(i6+i9) +.885(i7+i8)
// I12= i1 +.823(r2-r13) +.935(r3-r12) +.239(r4-r11) -.663(r5-r10) -.993(r6-r9) -.465(r7-r8)  +.568(i2+i13) -.355(i3+i12) -.971(i4+i11) -.749(i5+i10) +.121(i6+i9) +.885(i7+i8)
// I4 = i1 -.993(r2-r13) -.239(r3-r12) +.935(r4-r11) +.465(r5-r10) -.823(r6-r9) -.663(r7-r8)  +.121(i2+i13) -.971(i3+i12) -.355(i4+i11) +.885(i5+i10) +.568(i6+i9) -.749(i7+i8)
// I11= i1 +.993(r2-r13) +.239(r3-r12) -.935(r4-r11) -.465(r5-r10) +.823(r6-r9) +.663(r7-r8)  +.121(i2+i13) -.971(i3+i12) -.355(i4+i11) +.885(i5+i10) +.568(i6+i9) -.749(i7+i8)
// I5 = i1 -.935(r2-r13) +.663(r3-r12) +.465(r4-r11) -.993(r5-r10) +.239(r6-r9) +.823(r7-r8)  -.355(i2+i13) -.749(i3+i12) +.885(i4+i11) +.121(i5+i10) -.971(i6+i9) +.568(i7+i8)
// I10= i1 +.935(r2-r13) -.663(r3-r12) -.465(r4-r11) +.993(r5-r10) -.239(r6-r9) -.823(r7-r8)  -.355(i2+i13) -.749(i3+i12) +.885(i4+i11) +.121(i5+i10) -.971(i6+i9) +.568(i7+i8)
// I6 = i1 -.663(r2-r13) +.993(r3-r12) -.823(r4-r11) +.239(r5-r10) +.465(r6-r9) -.993(r7-r8)  -.749(i2+i13) +.121(i3+i12) +.568(i4+i11) -.971(i5+i10) +.885(i6+i9) -.355(i7+i8)
// I9 = i1 +.663(r2-r13) -.993(r3-r12) +.823(r4-r11) -.239(r5-r10) -.465(r6-r9) +.993(r7-r8)  -.749(i2+i13) +.121(i3+i12) +.568(i4+i11) -.971(i5+i10) +.885(i6+i9) -.355(i7+i8)
// I7 = i1 -.239(r2-r13) +.465(r3-r12) -.663(r4-r11) +.823(r5-r10) -.935(r6-r9) +.935(r7-r8)  -.971(i2+i13) +.885(i3+i12) -.749(i4+i11) +.568(i5+i10) -.355(i6+i9) +.121(i7+i8)
// I8 = i1 +.239(r2-r13) -.465(r3-r12) +.663(r4-r11) -.823(r5-r10) +.935(r6-r9) -.935(r7-r8)  -.971(i2+i13) +.885(i3+i12) -.749(i4+i11) +.568(i5+i10) -.355(i6+i9) +.121(i7+i8)

void fft13(T2 *u) {
  const double
      COS1      =  0.88545602565320991,
      COS2      =  0.56806474673115581,
      COS3      =  0.12053668025532305,
      COS4      = -0.35460488704253562,
      COS5      = -0.74851074817110108,
      COS6      = -0.97094181742605201,
      SIN1      =  0.46472317204376856,
      SIN2_SIN1 =  1.7709120513064198,
      SIN3_SIN1 =  2.1361294934623114,
      SIN4_SIN1 =  2.0119854118170659,
      SIN5_SIN1 =  1.4269197193772403,
      SIN6_SIN1 =  0.51496391547486375;

  /*
  const double COS1 = 0.8854560256532098959003755220151;	// cos(tau/13)
  const double COS2 = 0.56806474673115580251180755912752;	// cos(2*tau/13)
  const double COS3 = 0.12053668025532305334906768745254;	// cos(3*tau/13)
  const double COS4 = -0.35460488704253562596963789260002;	// cos(4*tau/13)
  const double COS5 = -0.74851074817110109863463059970135;	// cos(5*tau/13)
  const double COS6 = -0.97094181742605202715698227629379;	// cos(6*tau/13)
  const double SIN1 = 0.4647231720437685456560153351331;	// sin(tau/13)
  const double SIN2_SIN1 = 1.7709120513064197918007510440302;	// sin(2*tau/13) / sin(tau/13) = .823/.465
  const double SIN3_SIN1 = 2.136129493462311605023615118255;	// sin(3*tau/13) / sin(tau/13) = .993/.465
  const double SIN4_SIN1 = 2.0119854118170658984988864189353;	// sin(4*tau/13) / sin(tau/13) = .935/.465
  const double SIN5_SIN1 = 1.426919719377240353084339333055;	// sin(5*tau/13) / sin(tau/13) = .663/.465
  const double SIN6_SIN1 = 0.51496391547486370122962521953258;	// sin(6*tau/13) / sin(tau/13) = .239/.465
  */

  X2_mul_t4(u[1], u[12]);				// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[11]);				// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(u[3], u[10]);				// (r4+ i4+),  (i4- -r4-)
  X2_mul_t4(u[4], u[9]);				// (r5+ i5+),  (i5- -r5-)
  X2_mul_t4(u[5], u[8]);				// (r6+ i6+),  (i6- -r6-)
  X2_mul_t4(u[6], u[7]);				// (r7+ i7+),  (i7- -r7-)

  T2 tmp213a = fmaT2(COS1, u[1], u[0]);
  T2 tmp312a = fmaT2(COS2, u[1], u[0]);
  T2 tmp411a = fmaT2(COS3, u[1], u[0]);
  T2 tmp510a = fmaT2(COS4, u[1], u[0]);
  T2 tmp69a = fmaT2(COS5, u[1], u[0]);
  T2 tmp78a = fmaT2(COS6, u[1], u[0]);
  u[0] = u[0] + u[1];

  tmp213a = fmaT2(COS2, u[2], tmp213a);
  tmp312a = fmaT2(COS4, u[2], tmp312a);
  tmp411a = fmaT2(COS6, u[2], tmp411a);
  tmp510a = fmaT2(COS5, u[2], tmp510a);
  tmp69a = fmaT2(COS3, u[2], tmp69a);
  tmp78a = fmaT2(COS1, u[2], tmp78a);
  u[0] = u[0] + u[2];

  tmp213a = fmaT2(COS3, u[3], tmp213a);
  tmp312a = fmaT2(COS6, u[3], tmp312a);
  tmp411a = fmaT2(COS4, u[3], tmp411a);
  tmp510a = fmaT2(COS1, u[3], tmp510a);
  tmp69a = fmaT2(COS2, u[3], tmp69a);
  tmp78a = fmaT2(COS5, u[3], tmp78a);
  u[0] = u[0] + u[3];

  tmp213a = fmaT2(COS4, u[4], tmp213a);
  tmp312a = fmaT2(COS5, u[4], tmp312a);
  tmp411a = fmaT2(COS1, u[4], tmp411a);
  tmp510a = fmaT2(COS3, u[4], tmp510a);
  tmp69a = fmaT2(COS6, u[4], tmp69a);
  tmp78a = fmaT2(COS2, u[4], tmp78a);
  u[0] = u[0] + u[4];

  tmp213a = fmaT2(COS5, u[5], tmp213a);
  tmp312a = fmaT2(COS3, u[5], tmp312a);
  tmp411a = fmaT2(COS2, u[5], tmp411a);
  tmp510a = fmaT2(COS6, u[5], tmp510a);
  tmp69a = fmaT2(COS1, u[5], tmp69a);
  tmp78a = fmaT2(COS4, u[5], tmp78a);
  u[0] = u[0] + u[5];

  tmp213a = fmaT2(COS6, u[6], tmp213a);
  tmp312a = fmaT2(COS1, u[6], tmp312a);
  tmp411a = fmaT2(COS5, u[6], tmp411a);
  tmp510a = fmaT2(COS2, u[6], tmp510a);
  tmp69a = fmaT2(COS4, u[6], tmp69a);
  tmp78a = fmaT2(COS3, u[6], tmp78a);
  u[0] = u[0] + u[6];

  T2 tmp213b = fmaT2(SIN2_SIN1, u[11], u[12]);		// .823/.465
  T2 tmp312b = fmaT2(SIN2_SIN1, u[12], -u[7]);
  T2 tmp411b = fmaT2(SIN2_SIN1, u[8], -u[9]);
  T2 tmp510b = fmaT2(SIN2_SIN1, -u[7], -u[10]);
  T2 tmp69b = fmaT2(SIN2_SIN1, u[10], -u[8]);
  T2 tmp78b = fmaT2(SIN2_SIN1, -u[9], -u[11]);

  tmp213b = fmaT2(SIN3_SIN1, u[10], tmp213b);		// .993/.465
  tmp312b = fmaT2(SIN3_SIN1, -u[8], tmp312b);
  tmp411b = fmaT2(SIN3_SIN1, u[12], tmp411b);
  tmp510b = fmaT2(SIN3_SIN1, u[9], tmp510b);
  tmp69b = fmaT2(SIN3_SIN1, -u[11], tmp69b);
  tmp78b = fmaT2(SIN3_SIN1, -u[7], tmp78b);

  tmp213b = fmaT2(SIN4_SIN1, u[9], tmp213b);		// .935/.465
  tmp312b = fmaT2(SIN4_SIN1, u[11], tmp312b);
  tmp411b = fmaT2(SIN4_SIN1, -u[10], tmp411b);
  tmp510b = fmaT2(SIN4_SIN1, u[12], tmp510b);
  tmp69b = fmaT2(SIN4_SIN1, u[7], tmp69b);
  tmp78b = fmaT2(SIN4_SIN1, u[8], tmp78b);

  tmp213b = fmaT2(SIN5_SIN1, u[8], tmp213b);		// .663/.465
  tmp312b = fmaT2(SIN5_SIN1, -u[9], tmp312b);
  tmp411b = fmaT2(SIN5_SIN1, u[7], tmp411b);
  tmp510b = fmaT2(SIN5_SIN1, -u[11], tmp510b);
  tmp69b = fmaT2(SIN5_SIN1, u[12], tmp69b);
  tmp78b = fmaT2(SIN5_SIN1, u[10], tmp78b);

  tmp213b = fmaT2(SIN6_SIN1, u[7], tmp213b);		// .239/.465
  tmp312b = fmaT2(SIN6_SIN1, u[10], tmp312b);
  tmp411b = fmaT2(SIN6_SIN1, u[11], tmp411b);
  tmp510b = fmaT2(SIN6_SIN1, -u[8], tmp510b);
  tmp69b = fmaT2(SIN6_SIN1, -u[9], tmp69b);
  tmp78b = fmaT2(SIN6_SIN1, u[12], tmp78b);

  fma_addsub(u[1], u[12], SIN1, tmp213a, tmp213b);
  fma_addsub(u[2], u[11], SIN1, tmp312a, tmp312b);
  fma_addsub(u[3], u[10], SIN1, tmp411a, tmp411b);
  fma_addsub(u[4], u[9], SIN1, tmp510a, tmp510b);
  fma_addsub(u[5], u[8], SIN1, tmp69a, tmp69b);
  fma_addsub(u[6], u[7], SIN1, tmp78a, tmp78b);
}

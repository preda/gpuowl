
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

void fft7(T2 *u) {
  const double COS1 = 0.6234898018587335305;		// cos(tau/7)
  const double COS2 = -0.2225209339563144043;		// cos(2*tau/7)
  const double COS3 = -0.9009688679024191262;		// cos(3*tau/7)
  const double SIN1 = 0.781831482468029809;		// sin(tau/7)
  const double SIN2_SIN1 = 1.2469796037174670611;	// sin(2*tau/7) / sin(tau/7) = .975/.782
  const double SIN3_SIN1 = 0.5549581320873711914;	// sin(3*tau/7) / sin(tau/7) = .434/.782

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

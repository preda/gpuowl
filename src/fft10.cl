// See prime95's gwnum/zr10.mac file for more detailed explanation of the formulas below
//R1 = (r1+r6)     +((r2+r7)+(r5+r10))     +((r3+r8)+(r4+r9))
//R3 = (r1+r6) +.309((r2+r7)+(r5+r10)) -.809((r3+r8)+(r4+r9)) +.951((i2+i7)-(i5+i10)) +.588((i3+i8)-(i4+i9))
//R9 = (r1+r6) +.309((r2+r7)+(r5+r10)) -.809((r3+r8)+(r4+r9)) -.951((i2+i7)-(i5+i10)) -.588((i3+i8)-(i4+i9))
//R5 = (r1+r6) -.809((r2+r7)+(r5+r10)) +.309((r3+r8)+(r4+r9)) +.588((i2+i7)-(i5+i10)) -.951((i3+i8)-(i4+i9))
//R7 = (r1+r6) -.809((r2+r7)+(r5+r10)) +.309((r3+r8)+(r4+r9)) -.588((i2+i7)-(i5+i10)) +.951((i3+i8)-(i4+i9))
//R6 = (r1-r6)     -((r2-r7)-(r5-r10))     +((r3-r8)-(r4-r9))
//R2 = (r1-r6) +.809((r2-r7)-(r5-r10)) +.309((r3-r8)-(r4-r9)) +.588((i2-i7)+(i5-i10)) +.951((i3-i8)+(i4-i9))
//R10= (r1-r6) +.809((r2-r7)-(r5-r10)) +.309((r3-r8)-(r4-r9)) -.588((i2-i7)+(i5-i10)) -.951((i3-i8)+(i4-i9))
//R4 = (r1-r6) -.309((r2-r7)-(r5-r10)) -.809((r3-r8)-(r4-r9)) +.951((i2-i7)+(i5-i10)) -.588((i3-i8)+(i4-i9))
//R8 = (r1-r6) -.309((r2-r7)-(r5-r10)) -.809((r3-r8)-(r4-r9)) -.951((i2-i7)+(i5-i10)) +.588((i3-i8)+(i4-i9))

//I1 = (i1+i6)     +((i2+i7)+(i5+i10))     +((i3+i8)+(i4+i9))
//I3 = (i1+i6) +.309((i2+i7)+(i5+i10)) -.809((i3+i8)+(i4+i9)) -.951((r2+r7)-(r5+r10)) -.588((r3+r8)-(r4+r9))
//I9 = (i1+i6) +.309((i2+i7)+(i5+i10)) -.809((i3+i8)+(i4+i9)) +.951((r2+r7)-(r5+r10)) +.588((r3+r8)-(r4+r9))
//I5 = (i1+i6) -.809((i2+i7)+(i5+i10)) +.309((i3+i8)+(i4+i9)) -.588((r2+r7)-(r5+r10)) +.951((r3+r8)-(r4+r9))
//I7 = (i1+i6) -.809((i2+i7)+(i5+i10)) +.309((i3+i8)+(i4+i9)) +.588((r2+r7)-(r5+r10)) -.951((r3+r8)-(r4+r9))
//I6 = (i1-i6)     -((i2-i7)-(i5-i10))     +((i3-i8)-(i4-i9))
//I2 = (i1-i6) +.809((i2-i7)-(i5-i10)) +.309((i3-i8)-(i4-i9)) -.588((r2-r7)+(r5-r10)) -.951((r3-r8)+(r4-r9))
//I10= (i1-i6) +.809((i2-i7)-(i5-i10)) +.309((i3-i8)-(i4-i9)) +.588((r2-r7)+(r5-r10)) +.951((r3-r8)+(r4-r9))
//I4 = (i1-i6) -.309((i2-i7)-(i5-i10)) -.809((i3-i8)-(i4-i9)) -.951((r2-r7)+(r5-r10)) +.588((r3-r8)+(r4-r9))
//I8 = (i1-i6) -.309((i2-i7)-(i5-i10)) -.809((i3-i8)-(i4-i9)) +.951((r2-r7)+(r5-r10)) -.588((r3-r8)+(r4-r9))

void fft10(T2 *u) {
  const double SIN1 = 0x1.e6f0e134454ffp-1;		// sin(tau/5), 0.95105651629515353118
  const double SIN2_SIN1 = 0.618033988749894848;	// sin(2*tau/5) / sin(tau/5) = .588/.951, 0.618033988749894848
  const double COS1 = 0.309016994374947424;		// cos(tau/5), 0.309016994374947424
  const double COS2 = -0.809016994374947424;		// cos(2*tau/5), 0.809016994374947424

  X2(u[0], u[5]);					// (r1+ i1+),  (r1-  i1-)
  X2_mul_t4(u[1], u[6]);				// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[7]);				// (r3+ i3+),  (i3- -r3-)
  X2_mul_t4(u[3], u[8]);				// (r4+ i4+),  (i4- -r4-)
  X2_mul_t4(u[4], u[9]);				// (r5+ i5+),  (i5- -r5-)

  X2_mul_t4(u[1], u[4]);				// (r2++  i2++),  (i2+- -r2+-)
  X2_mul_t4(u[2], u[3]);				// (r3++  i3++),  (i3+- -r3+-)
  X2_mul_t4(u[6], u[9]);				// (i2-+ -r2-+), (-r2-- -i2--)
  X2_mul_t4(u[7], u[8]);				// (i3-+ -r3-+), (-r3-- -i3--)

  T2 tmp39a = fmaT2(COS1, u[1], u[0]);
  T2 tmp57a = fmaT2(COS2, u[1], u[0]);
  u[0] = u[0] + u[1];
  T2 tmp210a = fmaT2(COS2, u[9], u[5]);
  T2 tmp48a = fmaT2(COS1, u[9], u[5]);
  u[5] = u[5] + u[9];

  tmp39a = fmaT2(COS2, u[2], tmp39a);
  tmp57a = fmaT2(COS1, u[2], tmp57a);
  u[0] = u[0] + u[2];
  tmp210a = fmaT2(COS1, -u[8], tmp210a);
  tmp48a = fmaT2(COS2, -u[8], tmp48a);
  u[5] = u[5] - u[8];

  T2 tmp39b = fmaT2(SIN2_SIN1, u[3], u[4]);		// (i2+- +.588/.951*i3+-, -r2+- -.588/.951*r3+-)
  T2 tmp57b = fmaT2(SIN2_SIN1, u[4], -u[3]);		// (.588/.951*i2+- -i3+-, -.588/.951*r2+- +r3+-)
  T2 tmp210b = fmaT2(SIN2_SIN1, u[6], u[7]);		// (.588/.951*i2-+ +i3-+, -.588/.951*r2-+ -r3-+)
  T2 tmp48b = fmaT2(SIN2_SIN1, -u[7], u[6]);		// (i2-+ -.588/.951*i3-+, -r2-+ +.588/.951*r3-+)

  fma_addsub(u[1], u[9], SIN1, tmp210a, tmp210b);
  fma_addsub(u[2], u[8], SIN1, tmp39a, tmp39b);
  fma_addsub(u[3], u[7], SIN1, tmp48a, tmp48b);
  fma_addsub(u[4], u[6], SIN1, tmp57a, tmp57b);
}

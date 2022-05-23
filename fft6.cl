
void fft6(T2 *u) {
  const double COS1 = -0.5;					                  // cos(tau/3) == -0.5
  const double SIN1 = 0.86602540378443864676372317075294;	// sin(tau/3) == sqrt(3)/2

  X2(u[0], u[3]);						// (r1+ i1+),  (r1-  i1-)
  X2_mul_t4(u[1], u[4]);					// (r2+ i2+),  (i2- -r2-)
  X2_mul_t4(u[2], u[5]);					// (r3+ i3+),  (i3- -r3-)

  X2_mul_t4(u[1], u[2]);					// (r2++  i2++),  (i2+- -r2+-)
  X2_mul_t4(u[4], u[5]);					// (i2-+ -r2-+), (-r2-- -i2--)

  T2 tmp35a = fmaT2(COS1, u[1], u[0]);
  u[0] = u[0] + u[1];
  T2 tmp26a = fmaT2(COS1, u[5], u[3]);
  u[3] = u[3] + u[5];

  fma_addsub(u[1], u[5], SIN1, tmp26a, u[4]);
  fma_addsub(u[2], u[4], SIN1, tmp35a, u[2]);
}


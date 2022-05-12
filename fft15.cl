
// 5 complex FFT where second though fifth inputs need to be multiplied by SIN1, and third input needs to multiplied by SIN2
void fft5delayedSIN1234(T2 *u) {
  const double SIN4_SIN1 = 2.44512490403509663921;		// sin(4*tau/15) / sin(tau/15) = .985/.643
  const double SIN3_SIN2 = 1.27977277603217842055;		// sin(3*tau/15) / sin(2*tau/15) = .985/.643
  const double COS1SIN1 = 0.12568853494543955095;		// cos(tau/5) * sin(tau/15) = .309 * .407
  const double COS1SIN2 = 0.2296443803543192195;		// cos(tau/5) * sin(2*tau/15) = .309 * .743
  const double COS2SIN1 = -0.32905685648333965483;		// cos(2*tau/5) * sin(tau/15) = -.809 * .407
  const double COS2SIN2 = -0.60121679309301633701;		// cos(2*tau/5) * sin(2*tau/15) = -.809 * .743
  const double SIN1 = 0.40673664307580020775;			// sin(tau/15) = .407
  const double SIN2 = 0.74314482547739423501;			// sin(2*tau/15) = .743
  const double SIN2_SIN1SIN1_SIN2 = 0.33826121271771642765;	// sin(2*tau/5) / sin(tau/5) * sin(tau/15) / sin(2*tau/15) = .588/.951 * .407/.743
  const double SIN2_SIN1SIN2_SIN1 = 1.12920428618240948485;	// sin(2*tau/5) / sin(tau/5) * sin(2*tau/15) / sin(tau/15) = .588/.951 * .743/.407
  const double SIN1SIN1 = 0.38682953481325584261;		// sin(tau/5) * sin(tau/15) = .951 * .407
  const double SIN1SIN2 = 0.70677272882130044775;		// sin(tau/5) * sin(2*tau/15) = .951 * .743

  fma_addsub_mul_t4(u[1], u[4], SIN4_SIN1, u[1], u[4]);		// (r2+ i2+),  (i2- -r2-)		we owe results a mul by SIN1
  fma_addsub_mul_t4(u[2], u[3], SIN3_SIN2, u[2], u[3]);		// (r3+ i3+),  (i3- -r3-)		we owe results a mul by SIN2

  T2 tmp25a = fmaT2(COS1SIN1, u[1], u[0]);
  T2 tmp34a = fmaT2(COS2SIN1, u[1], u[0]);
  u[0] = u[0] + SIN1 * u[1];

  tmp25a = fmaT2(COS2SIN2, u[2], tmp25a);
  tmp34a = fmaT2(COS1SIN2, u[2], tmp34a);
  u[0] = u[0] + SIN2 * u[2];

  T2 tmp25b = fmaT2(SIN2_SIN1SIN2_SIN1, u[3], u[4]);		// (i2- +.588/.951*i3-, -r2- -.588/.951*r3-)	we owe results a mul by .951*SIN1
  T2 tmp34b = fmaT2(SIN2_SIN1SIN1_SIN2, u[4], -u[3]);		// (.588/.951*i2- -i3-, -.588/.951*r2- +r3-)	we owe results a mul by .951*SIN2

  fma_addsub(u[1], u[4], SIN1SIN1, tmp25a, tmp25b);
  fma_addsub(u[2], u[3], SIN1SIN2, tmp34a, tmp34b);
}

// This version is faster (fewer F64 ops), but slightly less accurate
void fft15(T2 *u) {
  const double COS1_SIN1 = 2.24603677390421605416;	// cos(tau/15) / sin(tau/15) = .766/.643
  const double COS2_SIN2 = 0.90040404429783994512;	// cos(2*tau/15) / sin(2*tau/15) = .174/.985
  const double COS3_SIN3 = 0.32491969623290632616;	// cos(3*tau/15) / sin(3*tau/15) = .174/.985
  const double COS4_SIN4 = -0.10510423526567646251;	// cos(4*tau/15) / sin(4*tau/15) = .174/.985

  fft3by(u, 5);
  fft3by(u+1, 5);
  fft3by(u+2, 5);
  fft3by(u+3, 5);
  fft3by(u+4, 5);

  u[6] = partial_cmul(u[6], COS1_SIN1);			// mul by w^1, we owe result a mul by SIN1
  u[11] = partial_cmul_conjugate(u[11], COS1_SIN1);	// mul by w^-1, we owe result a mul by SIN1
  u[7] = partial_cmul(u[7], COS2_SIN2);			// mul by w^2, we owe result a mul by SIN2
  u[12] = partial_cmul_conjugate(u[12], COS2_SIN2);	// mul by w^-2, we owe result a mul by SIN2
  u[8] = partial_cmul(u[8], COS3_SIN3);			// mul by w^3, we owe result a mul by SIN3
  u[13] = partial_cmul_conjugate(u[13], COS3_SIN3);	// mul by w^-3, we owe result a mul by SIN3
  u[9] = partial_cmul(u[9], COS4_SIN4);			// mul by w^4, we owe result a mul by SIN4
  u[14] = partial_cmul_conjugate(u[14], COS4_SIN4);	// mul by w^-4, we owe result a mul by SIN4

  fft5(u);
  fft5delayedSIN1234(u+5);
  fft5delayedSIN1234(u+10);

  // fix order [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 14, 2, 5, 8, 11]

  T2 tmp = u[1];
  u[1] = u[5];
  u[5] = u[12];
  u[12] = u[4];
  u[4] = u[6];
  u[6] = u[2];
  u[2] = u[11];
  u[11] = u[14];
  u[14] = u[10];
  u[10] = u[8];
  u[8] = u[13];
  u[13] = u[9];
  u[9] = u[3];
  u[3] = tmp;
}

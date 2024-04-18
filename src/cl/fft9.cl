// Copyright (C) Mihai Preda and George Woltman

#include "fft3.cl"

#if !NEW_FFT9 && !OLD_FFT9
#define NEW_FFT9 1
#endif

#if OLD_FFT9
// Adapted from: Nussbaumer, "Fast Fourier Transform and Convolution Algorithms", 5.5.7 "9-Point DFT".
void fft9(T2 *u) {
  const double C0 = 0x1.8836fa2cf5039p-1; //   0.766044443118978013 (2*c(u) - c(2*u) - c(4*u))/3
  const double C1 = 0x1.e11f642522d1cp-1; //   0.939692620785908428 (c(u) + c(2*u) - 2*c(4*u))/3
  const double C2 = 0x1.63a1a7e0b738ap-3; //   0.173648177666930359 -(c(u) - 2*c(2*u) + c(4*u))/3
  const double C3 = 0x1.bb67ae8584caap-1; //   0.866025403784438597 s(3*u)
  const double C4 = 0x1.491b7523c161dp-1; //   0.642787609686539363 s(u)
  const double C5 = 0x1.5e3a8748a0bf5p-2; //   0.342020143325668713 s(4*u)
  const double C6 = 0x1.f838b8c811c17p-1; //   0.984807753012208020 s(2*u)

  X2(u[1], u[8]);
  X2(u[2], u[7]);
  X2(u[3], u[6]);
  X2(u[4], u[5]);

  T2 m4 = (u[2] - u[4]) * C1;
  T2 s0 = (u[2] - u[1]) * C0 - m4;

  X2(u[1], u[4]);

  T2 t5 = u[1] + u[2];

  T2 m8  = mul_t4(u[7] + u[8]) * C4;
  T2 m10 = mul_t4(u[5] - u[8]) * C6;

  X2(u[5], u[7]);

  T2 m9  = mul_t4(u[5]) * C5;
  T2 t10 = u[8] + u[7];

  T2 s2 = m8 + m9;
  u[5] = m9 - m10;

  u[2] = u[0] - u[3] / 2;
  u[0] += u[3];
  u[3] = u[0] - t5 / 2;
  u[0] += t5;

  u[7] = mul_t4(u[6]) * C3;
  u[8] = u[7] + s2;
  u[6] = mul_t4(t10)  * C3;

  u[1] = u[2] - s0;

  u[4] = u[4] * C2 - m4;

  X2(u[2], u[4]);

  u[4] += s0;

  X2(u[5], u[7]);
  u[5] -= s2;

  X2(u[4], u[5]);
  X2(u[3], u[6]);
  X2(u[2], u[7]);
  X2(u[1], u[8]);
}

#elif NEW_FFT9

// 3 complex FFT where second input needs to be multiplied by SIN1 and third input needs to multiplied by SIN2
void fft3delayedSIN12(T2 *u) {
  const double SIN2_SIN1 = 1.5320888862379560704047853011108;	// sin(2*tau/9) / sin(tau/9) = .985/.643
  const double COS1SIN1 = -0.32139380484326966316132170495363;	// cos(tau/3) * sin(tau/9) = -.5 * .643
  const double SIN1SIN1 = 0.55667039922641936645291295204702;	// sin(tau/3) * sin(tau/9) = .866 * .643
  const double SIN1 = 0.64278760968653932632264340990726;	// sin(tau/9) = .643
  fma_addsub(u[1], u[2], SIN2_SIN1, u[1], u[2]);		// (r2+r3 i2+i3),  (i2-i3 -(r2-r3))	we owe results a mul by SIN1
  u[2] = mul_t4(u[2]);
  T2 tmp23 = u[0] + COS1SIN1 * u[1];
  u[0] = u[0] + SIN1 * u[1];
  fma_addsub (u[1], u[2], SIN1SIN1, tmp23, u[2]);
}

// This version is faster (fewer F64 ops), but slightly less accurate
void fft9(T2 *u) {
  const double COS1_SIN1 = 1.1917535925942099587053080718604;	// cos(tau/9) / sin(tau/9) = .766/.643
  const double COS2_SIN2 = 0.17632698070846497347109038686862;	// cos(2*tau/9) / sin(2*tau/9) = .174/.985

  fft3by(u, 3);
  fft3by(u+1, 3);
  fft3by(u+2, 3);

  u[4] = partial_cmul(u[4], COS1_SIN1);			// mul u[4] by w^1, we owe result a mul by SIN1
  u[7] = partial_cmul_conjugate(u[7], COS1_SIN1);	// mul u[7] by w^-1, we owe result a mul by SIN1
  u[5] = partial_cmul(u[5], COS2_SIN2);			// mul u[5] by w^2, we owe result a mul by SIN2
  u[8] = partial_cmul_conjugate(u[8], COS2_SIN2);	// mul u[8] by w^-2, we owe result a mul by SIN2

  fft3(u);
  fft3delayedSIN12(u+3);
  fft3delayedSIN12(u+6);

  // fix order [0, 3, 6, 1, 4, 7, 8, 2, 5]

  T2 tmp = u[1];
  u[1] = u[3];
  u[3] = tmp;
  tmp = u[2];
  u[2] = u[7];
  u[7] = u[5];
  u[5] = u[8];
  u[8] = u[6];
  u[6] = tmp;
}
#endif

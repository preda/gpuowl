// Copyright (C) Mihai Preda and George Woltman

// Adapted from: Nussbaumer, "Fast Fourier Transform and Convolution Algorithms", 5.5.7 "9-Point DFT".
// 12 FMA + 8 MUL, 72 ADD
void fft9(T2 *u) {
  double
      C0 = 0x1.8836fa2cf5039p-1, //   0.766044443118978013 (2*c(u) - c(2*u) - c(4*u))/3
      C1 = 0x1.e11f642522d1cp-1, //   0.939692620785908428 (c(u) + c(2*u) - 2*c(4*u))/3
      C2 = 0x1.63a1a7e0b738ap-3, //   0.173648177666930359 -(c(u) - 2*c(2*u) + c(4*u))/3
      C3 = 0x1.bb67ae8584caap-1, //   0.866025403784438597 s(3*u)
      C4 = 0x1.491b7523c161dp-1, //   0.642787609686539363 s(u)
      C5 = 0x1.5e3a8748a0bf5p-2, //   0.342020143325668713 s(4*u)
      C6 = 0x1.f838b8c811c17p-1; //   0.984807753012208020 s(2*u)

  X2(u[1], u[8]);
  X2(u[2], u[7]);
  X2(u[3], u[6]);
  X2(u[4], u[5]);

  T2 m4 = (u[4] - u[2]) * C1;
  T2 s0 = fmaT2(C0, u[2] - u[1], m4);

  X2(u[1], u[4]);

  T2 t5 = u[1] + u[2];

  T2 m8  = mul_t4(u[8] + u[7]);
  T2 m10 = mul_t4(u[8] - u[5]);

  X2(u[5], u[7]);

  T2 m9  = mul_t4(u[5]) * C5;
  T2 t10 = u[8] + u[7];

  T2 s2 = fmaT2(C4, m8, m9);
  u[5]  = fmaT2(C6, m10, m9);

  u[2] = fmaT2(-0.5, u[3], u[0]);
  u[0] += u[3];

  u[3] = fmaT2(-0.5, t5, u[0]);
  u[0] += t5;

  u[7] = mul_t4(u[6]) * C3;
  u[8] = u[7] + s2;
  u[6] = mul_t4(t10)  * C3;

  u[1] = u[2] - s0;

  u[4] = fmaT2(C2, u[4], m4);

  X2(u[2], u[4]);

  u[4] += s0;

  X2(u[5], u[7]);
  u[5] -= s2;

  X2(u[4], u[5]);
  X2(u[3], u[6]);
  X2(u[2], u[7]);
  X2(u[1], u[8]);
}

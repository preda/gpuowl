// Copyright (C) Mihai Preda

void fft3by(T2 *u, u32 base, u32 step, u32 m) {
#define A(k) u[(base + k * step) % m]
  // const double COS1 = -0.5;				  // cos(tau/3), -0.5
  const double SIN1 = 0.8660254037844386; // sin(tau/3) == sqrt(3)/2

  X2(A(1), A(2));
  A(2) = mul_t4(A(2));

  // T2 tmp23 = A(0) - A(1) / 2;
  T2 tmp23 = fmaT2(-0.5, A(1), A(0));

  A(0) = A(0) + A(1);
  A(1) = fmaT2( SIN1, A(2), tmp23);
  A(2) = fmaT2(-SIN1, A(2), tmp23);
#undef A
}

void fft3(T2 *u) { fft3by(u, 0, 1, 3); }

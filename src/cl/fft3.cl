// Copyright (C) Mihai Preda

#pragma once

// 6 FMA + 6 ADD
void fft3by(T2 *u, u32 base, u32 step, u32 M) {
#define A(k) u[(base + k * step) % M]
  double SIN1 = 0.8660254037844386; // sin(tau/3) == sqrt(3)/2

  X2(A(1), A(2));
  A(2) = mul_t4(A(2));

  T2 t = fmaT2(-0.5, A(1), A(0));

  A(0) = A(0) + A(1);

#if 1
  // here we prefer 2 FMAs vs. 1 MUL + 2 ADDs.
  A(1) = fmaT2( SIN1, A(2), t);
  A(2) = fmaT2(-SIN1, A(2), t);
#else
  T2 tmp = SIN1 * A(2);
  A(1) = t + tmp;
  A(2) = t - tmp;
#endif

#undef A
}

void fft3(T2 *u) { fft3by(u, 0, 1, 3); }

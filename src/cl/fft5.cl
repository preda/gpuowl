// Copyright (C) Mihai Preda and George Woltman

#pragma once

// Adapted from: Nussbaumer, "Fast Fourier Transform and Convolution Algorithms", 5.5.4 "5-Point DFT".
// 12 FMA + 24 ADD (or 10 FMA + 28 ADD)
void fft5by(T2 *u, u32 base, u32 step, u32 m) {
  const double
      S1 = -0.048943483704846427, // sin(tau/5) - 1
      S2 = 1.5388417685876268,    // sin(t/5) + sin(2t/5); S2 - 1 == 0.53884176858762667
      S3 = 0.36327126400268045,   // sin(t/5) - sin(2t/5)
      // C1 = 0.11803398874989485,// cos(t/5) - cos(2t/5) - 1
      C2 = 0.55901699437494745;   // (cos(t/5) - cos(2t/5)) / 2

#define A(k) u[(base + k * step) % m]

  X2(A(2), A(3));
  X2(A(1), A(4));
  X2(A(1), A(2));

  A(3) = mul_t4(A(3));
  A(4) = mul_t4(A(4));
  T2 t = A(4) - A(3);
  t = fmaT2(S1, t, t);
  A(3) = fmaT2( S2, A(3), t);
  A(4) = fmaT2(-S3, A(4), t);
  SWAP(A(3), A(4));

  t = A(0);
  A(0) = A(0) + A(1);
  A(1) = fmaT2(-0.25, A(1), t);

#if 0
  A(2) = C2 * A(2);
  X2(A(1), A(2));
#else
  t = A(1);
  A(1) = fmaT2( C2, A(2), t);
  A(2) = fmaT2(-C2, A(2), t);
#endif

  X2(A(1), A(4));
  X2(A(2), A(3));
#undef A
}

void fft5(T2 *u) { return fft5by(u, 0, 1, 5); }

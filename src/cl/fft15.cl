// Copyright (C) Mihai Preda and George Woltman

// The fft3by() and fft5by() below use a different "output map" relative to fft3.cl and fft5.cl
// This way fft15() does not need a "fix order" step at the end.
// See "An In-Place, In-Order Prime Factor Algorithm" by Burrus & Eschenbacher (1981)

// 6 FMA + 6 ADD
void fft3by(T2 *u, u32 base, u32 step, u32 M) {
#define A(k) u[(base + k * step) % M]
  double SIN1 = 0.8660254037844386; // sin(tau/3) == sqrt(3)/2

  X2(A(1), A(2));
  A(2) = mul_t4(A(2));

  T2 t = fmaT2(-0.5, A(1), A(0));

  A(0) = A(0) + A(1);

  // here we prefer 2 FMAs vs. 1 MUL + 2 ADDs.
  A(1) = fmaT2(-SIN1, A(2), t);
  A(2) = fmaT2( SIN1, A(2), t);
#undef A
}

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

  t = A(0);
  A(0) = A(0) + A(1);
  A(1) = fmaT2(-0.25, A(1), t);

  t = A(1);
  A(1) = fmaT2( C2, A(2), t);
  A(2) = fmaT2(-C2, A(2), t);

  X2(A(1), A(3));
  X2(A(2), A(4));

  t = A(1);
  A(1) = A(4);
  A(4) = A(2);
  A(2) = t;
#undef A
}

// FFT 15 using PFA ("Prime Factor Algorithm"): 66 FMA + 102 ADD
void fft15(T2 *u) {
  fft3by(u,  0, 5, 15);
  fft3by(u,  3, 5, 15);
  fft3by(u,  6, 5, 15);
  fft3by(u,  9, 5, 15);
  fft3by(u, 12, 5, 15);
  fft5by(u,  0, 3, 15);
  fft5by(u,  5, 3, 15);
  fft5by(u, 10, 3, 15);
}

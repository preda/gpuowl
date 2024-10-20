// Copyright (C) Mihai Preda

#pragma once

void fft4Core(T2 *u) {
  X2(u[0], u[2]);
  X2(u[1], u[3]); u[3] = mul_t4(u[3]);

  X2(u[0], u[1]);
  X2(u[2], u[3]);
}

void fft4by(T2 *u, u32 base, u32 step, u32 M) {

#define A(k) u[(base + step * k) % M]

#if 1
  double x0 = A(0).x + A(2).x;
  double x2 = A(0).x - A(2).x;
  double y0 = A(0).y + A(2).y;
  double y2 = A(0).y - A(2).y;

  double x1 = A(1).x + A(3).x;
  double y3 = A(1).x - A(3).x;
  double y1 = A(1).y + A(3).y;
  double x3 = -(A(1).y - A(3).y);

  double a0 = x0 + x1;
  double a1 = x0 - x1;

  double b0 = y0 + y1;
  double b1 = y0 - y1;

  double a2 = x2 + x3;
  double a3 = x2 - x3;

  double b2 = y2 + y3;
  double b3 = y2 - y3;

  A(0) = U2(a0, b0);
  A(1) = U2(a2, b2);
  A(2) = U2(a1, b1);
  A(3) = U2(a3, b3);

#else

  X2(A(0), A(2));
  X2(A(1), A(3));
  X2(A(0), A(1));

  A(3) = mul_t4(A(3));
  X2(A(2), A(3));
  SWAP(A(1), A(2));

#endif

#undef A

}

void fft4(T2 *u) { fft4by(u, 0, 1, 4); }

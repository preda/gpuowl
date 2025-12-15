// Copyright (C) Mihai Preda

#pragma once

#if FFT_FP64

void OVERLOAD fft4Core(T2 *u) {
  X2(u[0], u[2]);
  X2(u[1], u[3]); u[3] = mul_t4(u[3]);

  X2(u[0], u[1]);
  X2(u[2], u[3]);
}

// 16 ADD
void OVERLOAD fft4by(T2 *u, u32 base, u32 step, u32 M) {

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

void OVERLOAD fft4(T2 *u) { fft4by(u, 0, 1, 4); }

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

void OVERLOAD fft4Core(F2 *u) {
  X2(u[0], u[2]);
  X2(u[1], u[3]); u[3] = mul_t4(u[3]);

  X2(u[0], u[1]);
  X2(u[2], u[3]);
}

// 16 ADD
void OVERLOAD fft4by(F2 *u, u32 base, u32 step, u32 M) {

#define A(k) u[(base + step * k) % M]

#if 1
  float x0 = A(0).x + A(2).x;
  float x2 = A(0).x - A(2).x;
  float y0 = A(0).y + A(2).y;
  float y2 = A(0).y - A(2).y;

  float x1 = A(1).x + A(3).x;
  float y3 = A(1).x - A(3).x;
  float y1 = A(1).y + A(3).y;
  float x3 = -(A(1).y - A(3).y);

  float a0 = x0 + x1;
  float a1 = x0 - x1;

  float b0 = y0 + y1;
  float b1 = y0 - y1;

  float a2 = x2 + x3;
  float a3 = x2 - x3;

  float b2 = y2 + y3;
  float b3 = y2 - y3;

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

void OVERLOAD fft4(F2 *u) { fft4by(u, 0, 1, 4); }

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

void OVERLOAD fft4Core(GF31 *u) {
  X2(u[0], u[2]);
  X2_mul_t4(u[1], u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);
}

// 16 ADD
void OVERLOAD fft4by(GF31 *u, u32 base, u32 step, u32 M) {

#define A(k) u[(base + step * k) % M]

  Z31 x0 = add(A(0).x, A(2).x);                                 //GWBUG:  Delay some of the mods using 64 bit temps?
  Z31 x2 = sub(A(0).x, A(2).x);
  Z31 y0 = add(A(0).y, A(2).y);
  Z31 y2 = sub(A(0).y, A(2).y);

  Z31 x1 = add(A(1).x, A(3).x);
  Z31 y3 = sub(A(1).x, A(3).x);
  Z31 y1 = add(A(1).y, A(3).y);
  Z31 x3 = sub(A(3).y, A(1).y);

  Z31 a0 = add(x0, x1);
  Z31 a1 = sub(x0, x1);

  Z31 b0 = add(y0, y1);
  Z31 b1 = sub(y0, y1);

  Z31 a2 = add(x2, x3);
  Z31 a3 = sub(x2, x3);

  Z31 b2 = add(y2, y3);
  Z31 b3 = sub(y2, y3);

  A(0) = U2(a0, b0);
  A(1) = U2(a2, b2);
  A(2) = U2(a1, b1);
  A(3) = U2(a3, b3);

#undef A

}

void OVERLOAD fft4(GF31 *u) { fft4by(u, 0, 1, 4); }

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

void OVERLOAD fft4Core(GF61 *u) {       // Starts with all u[i] having maximum values of M61+epsilon.
  X2q(&u[0], &u[2], 2);                 // X2(u[0], u[2]);  No reductions mod M61.  Will require 3 M61s additions to make positives.
  X2q_mul_t4(&u[1], &u[3], 2);          // X2(u[1], u[3]); u[3] = mul_t4(u[3]);
  X2s(&u[0], &u[1], 3);
  X2s(&u[2], &u[3], 3);
}

// 16 ADD
void OVERLOAD fft4by(GF61 *u, u32 base, u32 step, u32 M) {

#define A(k) u[(base + step * k) % M]

  Z61 x0 = addq(A(0).x, A(2).x);             // Max value is 2*M61+epsilon
  Z61 x2 = subq(A(0).x, A(2).x, 2);          // Max value is 3*M61+epsilon
  Z61 y0 = addq(A(0).y, A(2).y);
  Z61 y2 = subq(A(0).y, A(2).y, 2);

  Z61 x1 = addq(A(1).x, A(3).x);
  Z61 y3 = subq(A(1).x, A(3).x, 2);
  Z61 y1 = addq(A(1).y, A(3).y);
  Z61 x3 = subq(A(3).y, A(1).y, 2);

  Z61 a0 = add(x0, x1);
  Z61 a1 = subs(x0, x1, 3);

  Z61 b0 = add(y0, y1);
  Z61 b1 = subs(y0, y1, 3);

  Z61 a2 = add(x2, x3);
  Z61 a3 = subs(x2, x3, 4);

  Z61 b2 = add(y2, y3);
  Z61 b3 = subs(y2, y3, 4);

  A(0) = U2(a0, b0);
  A(1) = U2(a2, b2);
  A(2) = U2(a1, b1);
  A(3) = U2(a3, b3);

#undef A

}

void OVERLOAD fft4(GF61 *u) { fft4by(u, 0, 1, 4); }

#endif

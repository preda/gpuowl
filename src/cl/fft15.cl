// Copyright (C) Mihai Preda and George Woltman

#include "fft3.cl"
#include "fft5.cl"

// FFT 15 using PFA ("Prime Factor Algorithm")
void fft15(T2 *u) {
  fft3by(u,  0, 5, 15);
  fft3by(u,  3, 5, 15);
  fft3by(u,  6, 5, 15);
  fft3by(u,  9, 5, 15);
  fft3by(u, 12, 5, 15);

  fft5by(u,  0, 3, 15);
  fft5by(u,  5, 3, 15);
  fft5by(u, 10, 3, 15);

  // Fix order [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13]

  T2 t = u[1];
  u[1] = u[8];
  u[8] = u[4];
  u[4] = u[2];
  u[2] = t;

  t = u[3];
  u[3]  = u[9];
  u[9]  = u[12];
  u[12] = u[6];
  u[6]  = t;


  SWAP(u[5], u[10]);

  t = u[7];
  u[7] = u[11];
  u[11] = u[13];
  u[13] = u[14];
  u[14] = t;
}

// Copyright (C) Mihai Preda and George Woltman

#include "fft3.cl"
#include "fft5.cl"

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

  // Fix order [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13]

#define CYCLE(a,b,c,d) { T2 t = u[a]; u[a] = u[b]; u[b] = u[c]; u[c] = u[d]; u[d] = t; }
  CYCLE(1, 8, 4, 2);
  CYCLE(3, 9, 12, 6);
  CYCLE(7, 11, 13, 14);
#undef CYCLE

  SWAP(u[5], u[10]);
}

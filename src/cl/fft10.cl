// Copyright (C) Mihai Preda

#include "fft5.cl"

// PFA(5*2): 24 FMA + 68 ADD
void fft10(T2 *u) {
  fft5by(u, 0, 2, 10);
  fft5by(u, 5, 2, 10);
  for (int i = 0; i < 10; i += 2) { X2(u[i], u[(i + 5) % 10]); }

  // Fix order: 0 3 6 9 2 5 8 1 4 7
#define CYCLE(a,b,c,d) { T2 t = u[a]; u[a] = u[b]; u[b] = u[c]; u[c] = u[d]; u[d] = t; }
  CYCLE(1, 7, 9, 3);
  CYCLE(2, 4, 8, 6);
#undef CYCLE
}

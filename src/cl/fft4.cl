// Copyright (C) Mihai Preda

#pragma once

void fft4Core(T2 *u) {
  X2(u[0], u[2]);
  X2(u[1], u[3]); u[3] = mul_t4(u[3]);

  X2(u[0], u[1]);
  X2(u[2], u[3]);
}

void fft4(T2 *u) {
   fft4Core(u);
   // revbin [0 2 1 3] undo
   SWAP(u[1], u[2]);
}

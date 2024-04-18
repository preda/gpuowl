// Copyright (C) Mihai Preda

#pragma once

void fft4Core(T2 *u) {
  X2(u[0], u[2]);
  X2(u[1], u[3]);
  X2(u[0], u[1]);

  T t = u[3].x;
  u[3].x = u[2].x - u[3].y;
  u[2].x = u[2].x + u[3].y;
  u[3].y = u[2].y + t;
  u[2].y = u[2].y - t;
}

void fft4(T2 *u) {
   fft4Core(u);
   // revbin [0 2 1 3] undo
   SWAP(u[1], u[2]);
}

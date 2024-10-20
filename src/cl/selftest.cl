// Copyright (C) Mihai Preda

#include "base.cl"
#include "trig.cl"
#include "fft4.cl"
#include "fft15.cl"
#include "fft14.cl"
#include "fft6.cl"
#include "fft9.cl"
#include "fft10.cl"
#include "fft11.cl"
#include "fft12.cl"

KERNEL(256) testFFT4(global double2* io) {
  T2 u[4];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 4; ++i) { u[i] = io[i]; }
    fft4(u);
    for (int i = 0; i < 4; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testTrig(global double2* out) {
  for (i32 k = get_global_id(0); k < ND / 8; k += get_global_size(0)) {
#if 0
    double angle = M_PI / (ND / 2) * k;
    out[k] = U2(cos(angle), -sin(angle));
#else
    out[k] = slowTrig_N(k, ND/8);
#endif
  }
}

KERNEL(256) testFFT(global double2* io) {
#define SIZE 12
  double2 u[SIZE];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < SIZE; ++i) { u[i] = io[i]; }
    fft12(u);
    for (int i = 0; i < SIZE; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT15(global double2* io) {
  double2 u[15];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 15; ++i) { u[i] = io[i]; }
    fft15(u);
    for (int i = 0; i < 15; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT14(global double2* io) {
  double2 u[14];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 14; ++i) { u[i] = io[i]; }
    fft14(u);
    for (int i = 0; i < 14; ++i) { io[i] = u[i]; }
  }
}

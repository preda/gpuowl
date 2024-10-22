// Copyright (C) Mihai Preda

#include "base.cl"
#include "trig.cl"
#include "fft3.cl"
#include "fft4.cl"
#include "fft5.cl"
#include "fft6.cl"
#include "fft7.cl"
#include "fft8.cl"
#include "fft9.cl"
#include "fft10.cl"
#include "fft11.cl"
#include "fft12.cl"
#include "fft13.cl"
#include "fft14.cl"
#include "fft15.cl"
#include "fft16.cl"

KERNEL(256) testFFT3(global double2* io) {
  T2 u[4];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 3; ++i) { u[i] = io[i]; }
    fft3(u);
    for (int i = 0; i < 3; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT4(global double2* io) {
  T2 u[4];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 4; ++i) { u[i] = io[i]; }
    fft4(u);
    for (int i = 0; i < 4; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT5(global double2* io) {
  T2 u[5];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 5; ++i) { u[i] = io[i]; }
    fft5(u);
    for (int i = 0; i < 5; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT6(global double2* io) {
  T2 u[6];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 6; ++i) { u[i] = io[i]; }
    fft6(u);
    for (int i = 0; i < 6; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT7(global double2* io) {
  T2 u[7];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 7; ++i) { u[i] = io[i]; }
    fft7(u);
    for (int i = 0; i < 7; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT8(global double2* io) {
  T2 u[8];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 8; ++i) { u[i] = io[i]; }
    fft8(u);
    for (int i = 0; i < 8; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT9(global double2* io) {
  T2 u[9];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 9; ++i) { u[i] = io[i]; }
    fft9(u);
    for (int i = 0; i < 9; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT10(global double2* io) {
  T2 u[10];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 10; ++i) { u[i] = io[i]; }
    fft10(u);
    for (int i = 0; i < 10; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT11(global double2* io) {
  T2 u[11];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 11; ++i) { u[i] = io[i]; }
    fft11(u);
    for (int i = 0; i < 11; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT13(global double2* io) {
  T2 u[13];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 13; ++i) { u[i] = io[i]; }
    fft13(u);
    for (int i = 0; i < 13; ++i) { io[i] = u[i]; }
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
#define SIZE 16
  double2 u[SIZE];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < SIZE; ++i) { u[i] = io[i]; }
    fft16(u);
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

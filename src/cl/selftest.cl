// Copyright (C) Mihai Preda

#include "base.cl"
#include "trig.cl"
#include "fft4.cl"
#include "fft15.cl"
#include "fft14.cl"
#include "fft6.cl"

KERNEL(256) testFFT4(global double2* out) {
  if (get_global_id(0) == 0) {
    T2 x[4] = {U2(1, 2), U2(-3, 4), U2(5, 6), U2(7, -8)};
    fft4(x);
    out[0] = x[0];
    out[1] = x[1];
    out[2] = x[2];
    out[3] = x[3];
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
  double2 u[7];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 6; ++i) { u[i] = io[i]; }
    fft6(u);
    for (int i = 0; i < 6; ++i) { io[i] = u[i]; }
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

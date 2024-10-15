#include "base.cl"
#include "trig.cl"
#include "fft4.cl"
#include "fft15.cl"

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

KERNEL(256) testFFT15(global double2* out) {
  double2 u[15];
  if (get_global_id(0) == 0) {
    u[0] = U2(1, 2);
    u[1] = U2(-3, -4);
    u[2] = U2(1, 0);
    u[3] = U2(0, 1);
    u[4] = U2(-1, 2);

    u[5] = U2(0, -1);
    u[6] = U2(0, -2);
    u[7] = U2(5, 1);
    u[8] = U2(5, 2);
    u[9] = U2(-2, -1);

    u[10] = U2(0, 1);
    u[11] = U2(0, 2);
    u[12] = U2(0, 3);
    u[13] = U2(1, -1);
    u[14] = U2(1, -2);

    fft15(u);

    for (int i = 0; i < 15; ++i) { out[i] = u[i]; }
  }
}

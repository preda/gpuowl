// gpuOwL, an OpenCL Mersenne primality test.
// Copyright (C) 2017 Mihai Preda.

#include "base.cl"

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

KERNEL(256) fft1K(d2ptr io, const d2ptr trig1k) {
  local double lds[4 * 256];
  double2 u[4];
  fft(4, lds, u, io, trig1k);
}

KERNEL(256) fft2K(d2ptr io, const d2ptr trig2k) {
  local double lds[8 * 256];
  double2 u[8];
  fft(8, lds, u, io, trig2k);
}

KERNEL(256) fftPremul1K(const i2ptr in, d2ptr out, const d2ptr A, const d2ptr trig1k) {
  local double lds[4 * 256];
  double2 u[4];
  fftPremul(4, lds, u, in, out, A, trig1k);
}

KERNEL(256) fftPremul2K(const i2ptr in, d2ptr out, const d2ptr A, const d2ptr trig2k) {
  local double lds[8 * 256];
  double2 u[8];
  fftPremul(8, lds, u, in, out, A, trig2k);
}

KERNEL(256) carryConv1K_2K(uint baseBitlen, d2ptr io,
                           global double * restrict carryShuttle, volatile global uint * restrict ready,
                           const d2ptr A, const d2ptr iA, const d2ptr trig1k) {
  local double lds[4 * 256];
  double2 u[4];
  carryConvolution(4, 2048, lds, u, baseBitlen, io, carryShuttle, ready, A, iA, trig1k);
}

#ifdef ENABLE_BUG

KERNEL(256) carryConv2K_2K(uint baseBitlen, d2ptr io,
                           global double * restrict carryShuttle, volatile global uint * restrict ready,
                           const d2ptr A, const d2ptr iA, const d2ptr trig1k) {
  local double lds[8 * 256];
  double2 u[8];
  carryConvolution(8, 2048, lds, u, baseBitlen, io, carryShuttle, ready, A, iA, trig1k);
}

#endif

void reverse(local double2 *lds, double2 *u, bool bump) {
  uint me = get_local_id(0);
  uint rm = 255 - me + bump;
  
  bar();

  lds[rm + 0 * 256] = u[7];
  lds[rm + 1 * 256] = u[6];
  lds[rm + 2 * 256] = u[5];
  lds[bump ? ((rm + 3 * 256) & 1023) : (rm + 3 * 256)] = u[4];
  
  bar();
  for (int i = 0; i < 4; ++i) { u[4 + i] = lds[256 * i + me]; }
}

// This kernel is equivalent to the sequence: fft2K, csquare2K, fft2K.
// It does less global memory transfers, but uses more VGPRs.
KERNEL(256) tail(d2ptr io, const d2ptr trig, const d2ptr bigTrig) {
  uint g = get_group_id(0);
  uint me = get_local_id(0);
  local double lds[2048];
  
  double2 u[8];
  read(8, u, io, g * 2048);
  fftImpl(8, lds, u, trig);

  reverse((local double2 *) lds, u, g == 0);

  double2 v[8];
  uint line2 = g ? 1024 - g : 512;
  read(8, v, io, line2 * 2048);
  bar(); fftImpl(8, lds, v, trig);

  reverse((local double2 *) lds, v, false);
  
  if (g == 0) { for (int i = 0; i < 4; ++i) { S2(u[4 + i], v[4 + i]); } }

  double2 tt = bigTrig[4096 + 512 + g];
  for (int i = 0; i < 4; ++i) {
    double2 a = u[i];
    double2 b = conjugate(v[4 + i]);
    double2 t = swap(mul(tt, bigTrig[256 * i + me]));
    if (i == 0 && g == 0 && me == 0) {
      a = 4 * foo(a);
      b = 8 * sq(b);
    } else {
      X2(a, b);
      M(b, conjugate(t));
      X2(a, b);
      a = sq(a);
      b = sq(b);
      X2(a, b);
      M(b,  t);
      X2(a, b);
    }
    u[i]     = conjugate(a);
    v[4 + i] = b;
  }

  tt = bigTrig[4096 + 512 +  line2];
  for (int i = 0; i < 4; ++i) {
    double2 a = v[i];
    double2 b = conjugate(u[4 + i]);
    double2 t = swap(mul(tt, bigTrig[256 * i + me]));
    X2(a, b);
    M(b, conjugate(t));
    X2(a, b);
    a = sq(a);
    b = sq(b);
    X2(a, b);
    M(b,  t);
    X2(a, b);
    v[i]     = conjugate(a);
    u[4 + i] = b;
  }

  if (g == 0) { for (int i = 0; i < 4; ++i) { S2(u[4 + i], v[4 + i]); } }

  reverse((local double2 *) lds, u, g == 0);
  bar(); fftImpl(8, lds, u, trig);

  write(8, u, io, g * 2048);
  
  reverse((local double2 *) lds, v, false);
  bar(); fftImpl(8, lds, v, trig);

  write(8, v, io, line2 * 2048);
}

// Carry propagation. conjugates input.
KERNEL(256) carryA(const uint baseBits, const d2ptr in, const d2ptr A, i2ptr out, global long *carryOut) {
  carryMul(1, baseBits, in, A, out, carryOut);
}

// Carry propagation + MUL 3. conjugates input.
KERNEL(256) carryMul3(const uint baseBits, const d2ptr in, const d2ptr A, i2ptr out, global long *carryOut) {
  carryMul(3, baseBits, in, A, out, carryOut);
}

KERNEL(256) carryB_2K(const uint baseBits, i2ptr io, const global long * restrict carryIn, const d2ptr A) {
  carryBCore(2048, baseBits, io, carryIn, A);
}

KERNEL(256) csquare2K(d2ptr io, const d2ptr trig)  { csquare(2048, io, trig); }
KERNEL(256) cmul2K(d2ptr io, const d2ptr in, const d2ptr trig)  { cmul(2048, io, in, trig); }

KERNEL(256) transpose1K_2K(const d2ptr in, d2ptr out, const d2ptr trig) {
  local double lds[4096];
  transpose(1024, 2048, lds, in, out, trig);
}

KERNEL(256) transpose2K_1K(const d2ptr in, d2ptr out, const d2ptr trig) {
  local double lds[4096];
  transpose(2048, 1024, lds, in, out, trig);
}

KERNEL(256) transpose2K_2K(const d2ptr in, d2ptr out, const d2ptr trig) {
  local double lds[4096];
  transpose(2048, 2048, lds, in, out, trig);
}

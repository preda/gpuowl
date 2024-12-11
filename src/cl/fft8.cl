// Copyright (C) Mihai Preda

#pragma once

#include "fft4.cl"

T2 mul_t8_delayed(T2 a) { return U2(a.x - a.y, a.x + a.y); }
T2 mul_3t8_delayed(T2 a) { return U2(-(a.x + a.y), a.x - a.y); }
//#define X2_apply_delay(a, b) { T2 t = a; a = t + M_SQRT1_2 * b; b = t - M_SQRT1_2 * b; }
#define X2_apply_delay(a, b) { T2 t = a; a.x = fma(b.x, M_SQRT1_2, a.x); a.y = fma(b.y, M_SQRT1_2, a.y); b.x = fma(-M_SQRT1_2, b.x, t.x); b.y = fma(-M_SQRT1_2, b.y, t.y); }

void fft4CoreSpecial(T2 *u) {
  X2(u[0], u[2]);
  X2(u[1], u[3]); u[3] = mul_t4(u[3]);
  X2_apply_delay(u[0], u[1]);
  X2_apply_delay(u[2], u[3]);
}

void fft8Core(T2 *u) {
  X2(u[0], u[4]);
  X2(u[1], u[5]);   u[5] = mul_t8_delayed(u[5]);
  X2(u[2], u[6]);   u[6] = mul_t4(u[6]);
  X2(u[3], u[7]);   u[7] = mul_3t8_delayed(u[7]);
  fft4Core(u);
  fft4CoreSpecial(u + 4);
}

// 4 MUL + 52 ADD
void fft8(T2 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

// The first four X2 operations have already been done
void fft8initialX2done(T2 *u) {
  u[5] = mul_t8_delayed(u[5]);
  u[6] = mul_t4(u[6]);
  u[7] = mul_3t8_delayed(u[7]);
  fft4Core(u);
  fft4CoreSpecial(u + 4);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

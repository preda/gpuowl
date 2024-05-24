// Copyright (C) Mihai Preda

#include "fftbase.cl"

// See also: fftheight.cl

// 64x4
void fft256w(local T2 *lds, T2 *u, Trig trig) {
  UNROLL_WIDTH_CONTROL
  for (u32 s = 0; s <= 4; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul(64, lds, trig, u, 4, 1u << s);
  }
  fft4(u);
}

// 64x8
void fft512w(local T2 *lds, T2 *u, Trig trig) {
  UNROLL_WIDTH_CONTROL
  for (u32 s = 0; s <= 3; s += 3) {
    if (s) { bar(); }
    fft8(u);
    shuflAndMul(64, lds, trig, u, 8, 1u << s);
  }
  fft8(u);
}

// 256x4
void fft1Kw(local T2 *lds, T2 *u, Trig trig) {
  UNROLL_WIDTH_CONTROL
  for (i32 s = 0; s <= 6; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul(256, lds, trig, u, 4, 1u << s);
  }
  fft4(u);
}

void fft_NW(T2 *u) {
#if NW == 4
  fft4(u);
#elif NW == 8
  fft8(u);
#else
#error NW
#endif
}

// 4K == 512x8 or 1024x4
void fft4Kw(local T2 *lds, T2 *u, Trig trig) {
  UNROLL_WIDTH_CONTROL
  for (u32 s = 1; s < WIDTH / NW; s *= NW) {
    if (s > 1) { bar(); }
    fft_NW(u);
    shuflAndMul(WIDTH / NW, lds, trig, u, NW, s);
  }
  fft_NW(u);
}

void fft_WIDTH(local T2 *lds, T2 *u, Trig trig) {
#if WIDTH == 256
  fft256w(lds, u, trig);
#elif WIDTH == 512
  fft512w(lds, u, trig);
#elif WIDTH == 1024
  fft1Kw(lds, u, trig);
#elif WIDTH == 4096
  fft4Kw(lds, u, trig);
#else
#error unexpected WIDTH.
#endif
}

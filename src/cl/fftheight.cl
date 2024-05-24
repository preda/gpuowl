// Copyright (C) Mihai Preda

#include "base.cl"
#include "fftbase.cl"
#include "middle.cl"

u32 transPos(u32 k, u32 middle, u32 width) { return k / width + k % width * middle; }

void fft256h(local T2 *lds, T2 *u, Trig trig) {
  for (u32 s = 0; s <= 4; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul(64, lds, trig, u, 4, 1u << s);
  }
  fft4(u);
}

void fft512h(local T2 *lds, T2 *u, Trig trig) {
  for (u32 s = 0; s <= 3; s += 3) {
    if (s) { bar(); }
    fft8(u);
    shuflAndMul(64, lds, trig, u, 8, 1u << s);
  }
  fft8(u);
}

void fft1Kh(local T2 *lds, T2 *u, Trig trig) {
  for (i32 s = 0; s <= 6; s += 2) {
    if (s) { bar(); }
    fft4(u);
    shuflAndMul(256, lds, trig, u, 4, 1u << s);
  }
  fft4(u);
}

void fft_NH(T2 *u) {
#if NH == 4
  fft4(u);
#elif NH == 8
  fft8(u);
#else
#error NH
#endif
}

void fft4Kh(local T2 *lds, T2 *u, Trig trig) {
  for (u32 s = 1; s < SMALL_HEIGHT / NH; s *= NH) {
    if (s > 1) { bar(); }
    fft_NH(u);
    shuflAndMul(SMALL_HEIGHT / NH, lds, trig, u, NH, s);
  }
  fft_NH(u);
}

void fft_HEIGHT(local T2 *lds, T2 *u, Trig trig) {
#if SMALL_HEIGHT == 256
  fft256h(lds, u, trig);
#elif SMALL_HEIGHT == 512
  fft512h(lds, u, trig);
#elif SMALL_HEIGHT == 1024
  fft1Kh(lds, u, trig);
#elif SMALL_HEIGHT == 4096
  fft4Kh(lds, u, trig);
#else
#error unexpected SMALL_HEIGHT.
#endif
}

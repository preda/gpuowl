#include "gpuowl.cl"

void reverse8(local T2 *lds, T2 *u, bool bump) {
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

void reverse4(local T2 *lds, T2 *u, bool bump) {
  uint me = get_local_id(0);
  uint rm = 255 - me + bump;
  
  bar();

  lds[rm + 0 * 256] = u[3];
  lds[bump ? ((rm + 256) & 511) : (rm + 256)] = u[2];
  
  bar();
  u[2] = lds[me];
  u[3] = lds[me + 256];
}

void reverse(uint N, local T2 *lds, T2 *u, bool bump) {
  if (N == 4) {
    reverse4(lds, u, bump);
  } else {
    reverse8(lds, u, bump);
  }
}

void halfSq(uint N, T2 *u, T2 *v, T2 tt, const G T2 *bigTrig, bool special) {
  uint g = get_group_id(0);
  uint me = get_local_id(0);
  for (int i = 0; i < N / 2; ++i) {
    T2 a = u[i];
    T2 b = conjugate(v[N / 2 + i]);
    T2 t = swap(mul(tt, bigTrig[256 * i + me]));
    if (special && i == 0 && g == 0 && me == 0) {
      a = shl(foo(a), 2);
      b = shl(sq(b), 3);
    } else {
      X2(a, b);
      b = mul(b, conjugate(t));
      X2(a, b);
      a = sq(a);
      b = sq(b);
      X2(a, b);
      b = mul(b, t);
      X2(a, b);
    }
    u[i] = conjugate(a);
    v[N / 2 + i] = b;
  }
}

void convolution(uint N, uint H, local T *lds, T2 *u, T2 *v, G T2 *io, const G T2 *trig, const G T2 *bigTrig) {
  uint W = N * 256;
  uint g = get_group_id(0);
  uint me = get_local_id(0);
  
  read(256, N, u, io, g * W);
  fftImpl(N, lds, u, trig);
  reverse(N, (local T2 *) lds, u, g == 0);
  
  uint line2 = g ? H - g : (H / 2);
  read(256, N, v, io, line2 * W);
  bar();
  fftImpl(N, lds, v, trig);
  reverse(N, (local T2 *) lds, v, false);
  
  if (g == 0) { for (int i = N / 2; i < N; ++i) { SWAP(u[i], v[i]); } }

  halfSq(N, u, v, bigTrig[W * 2 + (H / 2) + g],     bigTrig, true);
  
  halfSq(N, v, u, bigTrig[W * 2 + (H / 2) + line2], bigTrig, false);

  if (g == 0) { for (int i = N / 2; i < N; ++i) { SWAP(u[i], v[i]); } }

  reverse(N, (local T2 *) lds, u, g == 0);
  reverse(N, (local T2 *) lds, v, false);
  
  bar();
  fftImpl(N, lds, u, trig);
  write(256, N, u, io, g * W);
  
  bar();
  fftImpl(N, lds, v, trig);
  write(256, N, v, io, line2 * W);  
}

// "auto convolution" is equivalent to the sequence: fftH, square, fftH.
KERNEL(256) autoConv(P(T2) io, Trig smallTrig, P(T2) bigTrig) {
  local T lds[HEIGHT];
  T2 u[N_HEIGHT];
  T2 v[N_HEIGHT];
  convolution(N_HEIGHT, WIDTH, lds, u, v, io, smallTrig, bigTrig);
}

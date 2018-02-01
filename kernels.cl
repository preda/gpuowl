#include "gpuowl.cl"

KERNEL(256) fftW(P(T2) io, Trig smallTrig) {
  local T lds[WIDTH];
  T2 u[N_WIDTH];
  // fft(N_WIDTH, lds, u, io, smallTrig);

  uint g = get_group_id(0);
  uint step = g * WIDTH;
  io += step;

  read(256, N_WIDTH, u, io, 0);
  fftImpl(WIDTH, lds, u, smallTrig);
  write(256, N_WIDTH, u, io, 0);
}

KERNEL(256) fftH(P(T2) io, Trig smallTrig) {
  local T lds[HEIGHT];
  T2 u[N_HEIGHT];
  // fft(N_HEIGHT, lds, u, io, smallTrig);

  uint g = get_group_id(0);
  uint step = g * HEIGHT;
  io += step;

  read(256, N_HEIGHT, u, io, 0);
  fftImpl(HEIGHT, lds, u, smallTrig);
  write(256, N_HEIGHT, u, io, 0);

}

KERNEL(512) fftHTry(P(T2) io, Trig smallTrig) {
  local T lds[2048];
  T2 u[4];

  uint g = get_group_id(0);
  uint step = g * 2048;
  io += step;

  read(512, 4, u, io, 0);
  fft2kTry(512, lds, u, smallTrig);
  write(512, 4, u, io, 0);
}

KERNEL(512) fft4K(P(T2) io, Trig smallTrig) {
  local T lds[4096];
  T2 u[8];

  uint g = get_group_id(0);
  uint step = g * 4096;
  io += step;

  read(512, 8, u, io, 0);
  fft4kImpl(lds, u, smallTrig);
  write(512, 8, u, io, 0);
}

KERNEL(256) fftP(CP(Word2) in, P(T2) out, CP(T2) A, Trig smallTrig) {
  local T lds[WIDTH];
  T2 u[N_WIDTH];
  fftPremul(N_WIDTH, HEIGHT, lds, u, in, out, A, smallTrig);
}

KERNEL(256) carryA(CP(T2) in, CP(T2) A, P(Word2) out, P(Carry) carryOut) {
  carryACore(N_WIDTH, HEIGHT, 1, in, A, out, carryOut);
}

KERNEL(256) carryM(CP(T2) in, CP(T2) A, P(Word2) out, P(Carry) carryOut) {
  carryACore(N_WIDTH, HEIGHT, 3, in, A, out, carryOut);
}

KERNEL(256) carryB(P(Word2) io, CP(Carry) carryIn) {
  carryBCore(N_WIDTH, HEIGHT, io, carryIn);
}

KERNEL(256) square(P(T2) io, Trig bigTrig)  { csquare(HEIGHT, WIDTH, io, bigTrig); }

KERNEL(256) multiply(P(T2) io, CP(T2) in, Trig bigTrig)  { cmul(HEIGHT, WIDTH, io, in, bigTrig); }

// The "carryConvolution" is equivalent to the sequence: fftW, carryA, carryB, fftPremul.
// It uses "stairway" carry data forwarding from one group to the next.
// N gives the FFT size, W = N * 256.
// H gives the nuber of "lines" of FFT.
KERNEL(256) carryConv(P(T2) io, P(Carry) carryShuttle, volatile P(uint) ready,
                      CP(T2) A, CP(T2) iA, Trig smallTrig) {
  local T lds[WIDTH];

  uint gr = get_group_id(0);
  uint gm = gr % HEIGHT;
  uint me = get_local_id(0);
  uint step = gm * WIDTH;

  io    += step;
  A     += step;
  iA    += step;
  
  T2 u[N_WIDTH];
  read(256, N_WIDTH, u, io, 0);
  fftImpl(N_WIDTH * 256, lds, u, smallTrig);

  Word2 word[N_WIDTH];
  for (int i = 0; i < N_WIDTH; ++i) {
    uint p = i * 256 + me;
    uint pos = gm + HEIGHT * 256 * i + HEIGHT * me;
    Carry carry = 0;
    word[i] = unweightAndCarry(1, conjugate(u[i]), &carry, pos, iA, p);
    if (gr < HEIGHT) { carryShuttle[gr * WIDTH + p] = carry; }
  }

  bigBar();

  // Signal that this group is done writing the carry.
  if (gr < HEIGHT && me == 0) { atomic_xchg(&ready[gr], 1); }

  if (gr == 0) { return; }

  // Wait until the previous group is ready with the carry.
  if (me == 0) { while(!atomic_xchg(&ready[gr - 1], 0)); }

  bigBar();
  
  for (int i = 0; i < N_WIDTH; ++i) {
    uint p = i * 256 + me;
    uint pos = gm + HEIGHT * 256 * i + HEIGHT * me;
    Carry carry = carryShuttle[(gr - 1) * WIDTH + ((p - gr / HEIGHT) & (WIDTH - 1))];
    u[i] = carryAndWeightFinal(word[i], carry, pos, A, p);
  }

  fftImpl(N_WIDTH * 256, lds, u, smallTrig);
  write(256, N_WIDTH, u, io, 0);
}

KERNEL(256) transposeW(CP(T2) in, P(T2) out, Trig bigTrig) {
  local T lds[4096];
  transpose(WIDTH, HEIGHT, max(WIDTH, HEIGHT), lds, in, out, bigTrig);
}

KERNEL(256) transposeH(CP(T2) in, P(T2) out, Trig bigTrig) {
  local T lds[4096];
  transpose(HEIGHT, WIDTH, max(WIDTH, HEIGHT), lds, in, out, bigTrig);
}

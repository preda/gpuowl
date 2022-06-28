// GpuOwl Copyright (C) Mihai Preda

// --- carry ---

#define POW2(n) (((Word) 1) << (n))

Word carryStep(i64 *carry, u32 nBits) {
  i64 x = *carry;
  Word w = lowBits(x, nBits);

  if ((-w == POW2(nBits - 1)) && (x > 0)) {
    w = -w;
    assert(x >= w);
    assert(((x - w) & (POW2(nBits) - 1)) == 0);
  }

  *carry = (x - w) >> nBits;
  return w;
}

Word2 carryPair(T2 u, i64 *outCarry, uint2 nBits, i64 inCarry, Roundoff *maxROE) {
  *outCarry = inCarry + convert(u.x, maxROE);
  Word a = carryStep(outCarry, nBits.x);
  *outCarry += convert(u.y, maxROE);
  Word b = carryStep(outCarry, nBits.y);

#if 0
  assert(abs(a) <= POW2(nBits.x - 1));
  assert(abs(b) <= POW2(nBits.y - 1));

  if (a == -POW2(nBitsA - 1)) {
    if (b == -POW2(nBitsB - 1)) {
      b = -b;
      --*outCarry;
    }
    if (b > 0) {
      a = -a;
      --b;
    }
  } else if (a == POW2(nBitsA - 1)) {
    if (b == POW2(nBitsB - 1)) {
      b = -b;
      ++*outCarry;
    }
    if (b < 0) {
      a = -a;
      ++b;
    }
  }
#endif

  return (Word2) (a, b);
}

Word2 carryFinal(Word2 a, i64 inCarry, bool b1) {
  inCarry += a.x;
  // i64 tmpCarry = a.x + inCarry;
  a.x = carryStep(&inCarry, b1);
  a.y += inCarry;
  return a;
}

Word2 carryPairMul(T2 u, i64 *outCarry, uint2 nBits, i64 inCarry, Roundoff *maxROE) {
  *outCarry = 3 * convert(u.x, maxROE) + inCarry;
  Word a = carryStep(outCarry, nBits.x);
  *outCarry += 3 * convert(u.y, maxROE);
  Word b = carryStep(outCarry, nBits.y);
  return (Word2) (a, b);
}

// Carry propagation from word and carry.
Word2 carryWord64(Word2 a, i64 *carry, uint2 nBits) {
  *carry += a.x;
  a.x = carryStep(carry, nBits.x);
  *carry += a.y;
  a.y = carryStep(carry, nBits.y);
  return a;
}

// Propagate carry this many pairs of words.
#define CARRY_LEN 8

// computes 2*(a.x*b.x+a.y*b.y) + i*2*(a.x*b.y+a.y*b.x)
// which happens to be the cyclical convolution (a.x, a.y)x(b.x, b.y) * 2
T2 foo2(T2 a, T2 b) {
  a = addsub(a);
  b = addsub(b);
  return addsub(pair(mul(a.x, b.x), mul(a.y, b.y)));
}

// computes 2*[x^2+y^2 + i*(2*x*y)]. i.e. 2 * cyclical autoconvolution of (x, y)
T2 foo(T2 a) {
  a = addsub(a);
  return addsub(pair(sq(a.x), sq(a.y)));
}


#if AMDGPU
typedef constant const T2* Trig;
#else
typedef global const T2* Trig;
#endif

void fft4Core(T2 *u) {
  X2(u[0], u[2]);
  X2(u[1], u[3]); u[3] = mul_t4(u[3]);
  X2(u[0], u[1]);
  X2(u[2], u[3]);
}

void fft4(T2 *u) {
   fft4Core(u);
   // revbin [0 2 1 3] undo
   SWAP(u[1], u[2]);
}

void fft2(T2* u) {
  X2(u[0], u[1]);
}

void fft8Core(T2 *u) {
  X2(u[0], u[4]);
  X2(u[1], u[5]);   u[5] = mul_t8(u[5]);
  X2(u[2], u[6]);   u[6] = mul_t4(u[6]);
  X2(u[3], u[7]);   u[7] = mul_3t8(u[7]);
  fft4Core(u);
  fft4Core(u + 4);
}

void fft8(T2 *u) {
  fft8Core(u);
  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}


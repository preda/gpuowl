// Copyright (C) Mihai Preda

#pragma once

#include "base.cl"

// Multiply by 2 without using floating point instructions.  This is a little sloppy as an input of zero returns 2^-1022.
T OVERLOAD mul2(T a) { int2 tmp = as_int2(a); tmp.y += 0x00100000; /* Bump exponent by 1 */ return (as_double(tmp)); }
T2 OVERLOAD mul2(T2 a) { return U2(mul2(a.x), mul2(a.y)); }

// Multiply by -2 without using floating point instructions.  This is a little sloppy as an input of zero returns -2^-1022.
T OVERLOAD mulminus2(T a) { int2 tmp = as_int2(a); tmp.y += 0x80100000; /* Bump exponent by 1, flip sign bit */ return (as_double(tmp)); }
T2 OVERLOAD mulminus2(T2 a) { return U2(mulminus2(a.x), mulminus2(a.y)); }

// a * (b + 1) == a * b + a
T OVERLOAD fancyMul(T a, T b)   { return fma(a, b, a); }
T2 OVERLOAD fancyMul(T2 a, T2 b) { return U2(fancyMul(a.x, b.x), fancyMul(a.y, b.y)); }

T2 cmul(T2 a, T2 b) {
#if 1
  return U2(fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x));
#else
  return U2(fma(a.y, -b.y, a.x * b.x), fma(a.x, b.y, a.y * b.x));
#endif
}

T2 conjugate(T2 a) { return U2(a.x, -a.y); }

T2 cmul_by_conjugate(T2 a, T2 b) { return cmul(a, conjugate(b)); }

// Multiply a by b and conjugate(b).  This saves 2 multiplies.
void cmul_a_by_b_and_conjb(T2 *res1, T2 *res2, T2 a, T2 b) {
  T axbx = a.x * b.x;
  T aybx = a.y * b.x;
  res1->x = fma(a.y, -b.y, axbx), res1->y = fma(a.x,  b.y, aybx);
  res2->x = fma(a.y,  b.y, axbx), res2->y = fma(a.x, -b.y, aybx);
}

T2 cfma(T2 a, T2 b, T2 c) {
#if 1
  return U2(fma(a.x, b.x, fma(a.y, -b.y, c.x)), fma(a.y, b.x, fma(a.x, b.y, c.y)));
#else
  return U2(fma(a.y, -b.y, fma(a.x, b.x, c.x)), fma(a.x, b.y, fma(a.y, b.x, c.y)));
#endif
}

// Square any complex number
T2 csq(T2 a) { return U2(fma(a.x, a.x, - a.y * a.y), mul2(a.x) * a.y); }

// Square a (cos,sin) complex number.  Fancy squaring returns a fancy value.
T2 csqTrig(T2 a) { return U2(fma(mulminus2(a.y), a.y, 1), mul2(a.x) * a.y); }
T2 csqTrigFancy(T2 a) { return U2(mulminus2(a.y) * a.y, mul2(fma(a.x, a.y, a.y))); }

// Cube a complex number w (cos,sin) given w^2 and w.  The squared input can be either fancy or not fancy.
T2 ccubeTrig(T2 sq, T2 w) { T tmp = mul2(sq.y); return U2(fma(tmp, -w.y, w.x), fma(tmp, w.x, -w.y)); }
T2 ccubeTrigFancy(T2 sq, T2 w) { T tmp = mul2(sq.y); T wx = w.x + 1; return U2(fma(tmp, -w.y, wx), fma(tmp, wx, -w.y)); }

// a^2 + c
T2 csqa(T2 a, T2 c) { return U2(fma(a.x, a.x, fma(a.y, -a.y, c.x)), fma(mul2(a.x), a.y, c.y)); }

// Complex a * (b + 1)
// Useful for mul with twiddles of small angles, where the real part is stored with the -1 trick for increased precision
T2 cmulFancy(T2 a, T2 b) { return cfma(a, b, a); }

// Multiply a by fancy b and conjugate(fancy b).  This saves 2 FMAs.
void cmul_a_by_fancyb_and_conjfancyb(T2 *res1, T2 *res2, T2 a, T2 b) {
  T axbx = fma(a.x, b.x, a.x);
  T aybx = fma(a.y, b.x, a.y);
  res1->x = fma(a.y, -b.y, axbx), res1->y = fma(a.x,  b.y, aybx);
  res2->x = fma(a.y,  b.y, axbx), res2->y = fma(a.x, -b.y, aybx);
}

// Returns complex (a + 1) * (b + 1) - 1
T2 cmulFancyUpdate(T2 a, T2 b) { return cfma(a, b, a + b); }

// (a + 1)^2 - 1
T2 csqFancyUpdate(T2 a) {
  // Below we use (x + 1)^2 + y^2 == 1
  return mul2(U2(fma(a.x, a.x, mul2(a.x)), fma(a.x, a.y, a.y)));
}

T2 mul_t4(T2 a)  { return U2(-a.y, a.x); } // i.e. a * i

T2 mul_t8(T2 a)  { // mul(a, U2( 1, 1)) * (T)(M_SQRT1_2); }
  // One mul, two FMAs
  T ay = a.y * M_SQRT1_2;
  return U2(fma(a.x, M_SQRT1_2, -ay), fma(a.x, M_SQRT1_2, ay));
// Two adds, two muls
//  return U2(a.x - a.y, a.x + a.y) * M_SQRT1_2;
}

T2 mul_3t8(T2 a) { // mul(a, U2(-1, 1)) * (T)(M_SQRT1_2); }
  // One mul, two FMAs
  T ay = a.y * M_SQRT1_2;
  return U2(fma(-a.x, M_SQRT1_2, -ay), fma(a.x, M_SQRT1_2, -ay));
// Two adds, two muls
//  return U2(-(a.x + a.y), a.x - a.y) * M_SQRT1_2;
}

T2 swap(T2 a)   { return U2(a.y, a.x); }
T2 addsub(T2 a) { return U2(a.x + a.y, a.x - a.y); }

#define X2(a, b) { T2 t = a; a = t + b; b = t - b; }

// Same as X2(a, b), b = mul_t4(b)
#define X2_mul_t4(a, b) { X2(a, b); b = mul_t4(b); }
// { T2 t = a; a = a + b; t.x = t.x - b.x; b.x = b.y - t.y; b.y = t.x; }

// Same as X2(a, conjugate(b))
#define X2conjb(a, b) { T2 t = a; a.x = a.x + b.x; a.y = a.y - b.y; b.x = t.x - b.x; b.y = t.y + b.y; }

// Same as X2(a, b), a = conjugate(a)
#define X2conja(a, b) { T2 t = a; (a).x = (a).x + (b).x; (a).y = -(a).y - (b).y; b = t - b; }

#define SWAP(a, b) { T2 t = a; a = b; b = t; }

T2 fmaT2(T a, T2 b, T2 c) { return fma(U2(a, a), b, c); }

// a = c + sin * d; b = c - sin * d;
#define fma_addsub(a, b, sin, c, d) { T2 t = c + sin * d; b = c - sin * d; a = t; }

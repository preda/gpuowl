// Copyright (C) Mihai Preda

#pragma once

#include "base.cl"

// a * (b + 1) == a * b + a
OVERLOAD T  fancyMul(T a, T b)   { return fma(a, b, a); }
OVERLOAD T2 fancyMul(T2 a, T2 b) { return U2(fancyMul(a.x, b.x), fancyMul(a.y, b.y)); }

T2 cmul(T2 a, T2 b) {
#if 1
  return U2(fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x));
#else
  return U2(fma(a.y, -b.y, a.x * b.x), fma(a.x, b.y, a.y * b.x));
#endif
}

T2 conjugate(T2 a) { return U2(a.x, -a.y); }

T2 cmul_by_conjugate(T2 a, T2 b) { return cmul(a, conjugate(b)); }

T2 cfma(T2 a, T2 b, T2 c) {
#if 1
  return U2(fma(a.x, b.x, fma(a.y, -b.y, c.x)), fma(a.y, b.x, fma(a.x, b.y, c.y)));
#else
  return U2(fma(a.y, -b.y, fma(a.x, b.x, c.x)), fma(a.x, b.y, fma(a.y, b.x, c.y)));
#endif
}

T2 csq(T2 a) { return U2(fma(a.x, a.x, - a.y * a.y), 2 * a.x * a.y); }

// a^2 + c
T2 csqa(T2 a, T2 c) { return U2(fma(a.x, a.x, fma(a.y, -a.y, c.x)), fma(2 * a.x, a.y, c.y)); }


// Complex a * (b + 1)
// Useful for mul with twiddles of small angles, where the real part is stored with the -1 trick for increased precision
T2 cmulFancy(T2 a, T2 b) { return cfma(a, b, a); }

// Returns complex (a + 1) * (b + 1) - 1
T2 cmulFancyUpdate(T2 a, T2 b) { return cfma(a, b, a + b); }

// (a + 1)^2 - 1
T2 csqFancyUpdate(T2 a) {
  // Below we use (x + 1)^2 + y^2 == 1
#if 0
  return 2 * U2(-a.y * a.y, fma(a.x, a.y, a.y));
#else
  return 2 * U2(fma(a.x, a.x, 2 * a.x), fma(a.x, a.y, a.y));
#endif
}

T2 mul_t4(T2 a)  { return U2(-a.y, a.x); } // mul(a, U2( 0, 1)); }

T2 mul_t8(T2 a)  { // mul(a, U2( 1, 1)) * (T)(M_SQRT1_2); }
  return U2(a.x - a.y, a.x + a.y) * M_SQRT1_2;
}

T2 mul_3t8(T2 a) { // mul(a, U2(-1, 1)) * (T)(M_SQRT1_2); }
  return U2(a.y + a.x, a.y - a.x) * -M_SQRT1_2;
}

T2 swap(T2 a)   { return U2(a.y, a.x); }
T2 addsub(T2 a) { return U2(a.x + a.y, a.x - a.y); }

#define X2(a, b) { T2 t = a; a = t + b; b = t - b; }

// #define t4_X2(a, b) { T t = a.x; a.x = a.x - b.y; b.x = t + b.y; t = a.y; a.y = a.y + b.x; b.y = t - b.x; }

// Same as X2(a, b), b = mul_t4(b)
#define X2_mul_t4(a, b) { X2(a, b); b = mul_t4(b); }
// { T2 t = a; a = a + b; t.x = t.x - b.x; b.x = b.y - t.y; b.y = t.x; }

// Same as X2(a, conjugate(b))
#define X2conjb(a, b) { T2 t = a; a.x = a.x + b.x; a.y = a.y - b.y; b.x = t.x - b.x; b.y = t.y + b.y; }

// Same as X2(a, b), a = conjugate(a)
#define X2conja(a, b) { T2 t = a; a.x = a.x + b.x; a.y = -a.y - b.y; b = t - b; }

#define SWAP(a, b) { T2 t = a; a = b; b = t; }

T2 fmaT2(T a, T2 b, T2 c) { return fma(U2(a, a), b, c); }

// Partial complex multiplies:  the mul by sin is delayed so that it can be later propagated to an FMA instruction
// complex mul by cos-i*sin given cos/sin, sin
// T2 partial_cmul(T2 a, T c_over_s) { return U2(fma(a.x, c_over_s, a.y), fma(a.y, c_over_s, -a.x)); }
// complex mul by cos+i*sin given cos/sin, sin
// T2 partial_cmul_conjugate(T2 a, T c_over_s) { return U2(fma(a.x, c_over_s, -a.y), fma(a.y, c_over_s, a.x)); }

// a = c + sin * d; b = c - sin * d;
#define fma_addsub(a, b, sin, c, d) { T2 t = c + sin * d; b = c - sin * d; a = t; }

// Copyright (C) Mihai Preda

#pragma once

#include "base.cl"

T2 U2(T a, T b) { return (T2) (a, b); }

OVERLOAD T fancyMul(T x, const T y) {
  // x * (y + 1) == x * y + x
  return fma(x, y, x);
}

OVERLOAD T2 fancyMul(T2 x, const T2 y) {
  return U2(fancyMul(RE(x), RE(y)), fancyMul(IM(x), IM(y)));
}

// fma(x, y, z); }
// OVERLOAD double mad(double x, double y, double z) { return x * y + z; }

// complex fma
T2 cfma(T2 a, T2 b, T2 c) {
  return U2(fma(a.x, b.x, fma(a.y, -b.y, c.x)), fma(a.x, b.y, fma(a.y, b.x, c.y)));
}

// complex square
T2 sq(T2 a) { return U2(mad(a.x, a.x, - a.y * a.y), 2 * a.x * a.y); }

// a^2 + c
T2 sqa(T2 a, T2 c) { return U2(mad(a.x, a.x, mad(a.y, -a.y, c.x)), mad(2 * a.x, a.y, c.y)); }

// complex mul
OVERLOAD T2 mul(T2 a, T2 b) { return U2(mad(RE(a), RE(b), -IM(a)*IM(b)), mad(RE(a), IM(b), IM(a)*RE(b))); }

// Complex mul a * (b + 1)
// Useful for mul with twiddles of small angles, where the real part is stored with the -1 trick for increased precision
T2 fancyMulTrig(T2 a, T2 b) {
  return U2(
      #if 0
        fma(a.x, b.x, fma(a.y, -b.y, a.x)),
        fma(a.y, b.x, fma(a.x, b.y, a.y))
      #else
        fma(a.y, -b.y, fma(a.x, b.x, a.x)),
        fma(a.x,  b.y, fma(a.y, b.x, a.y))
      #endif
        );
}

T2 fancyMulTrigConj(T2 a, T2 b) {
  return U2(
        fma(a.y, b.y, fma(a.x, b.x, a.x)),
        fma(a.x, -b.y, fma(a.y, b.x, a.y))
        );
}

// Returns complex a * (b + 1) + c
T2 fancyMadTrig(T2 a, T2 b, T2 c) {
  return U2(
        fma(a.y, -b.y, fma(a.x, b.x, a.x + c.x)),
        fma(a.x, b.y, fma(a.y, b.x, a.y + c.y))
        );
}

// Returns complex (a + 1) * (b + 1) - 1 == a * (b + 1) + b
T2 fancyMulUpdate(T2 a, T2 b) { return fancyMadTrig(a, b, b); }

T2 fancySqUpdate(T2 a) {
  return 2 * U2(-a.y * a.y, fma(a.x, a.y, a.y));
  /*
      U2(
        // fma(a.y, -a.y, fma(a.x, a.x, 2 * a.x)),
        fma(a.x, a.x, fma(a.y, -a.y, 2 * a.x)),
        2 * fma(a.x, a.y, a.y)
        );
  */
}

T2 mul_t4(T2 a)  { return U2(IM(a), -RE(a)); } // mul(a, U2( 0, -1)); }

T2 mul_t8(T2 a)  { // mul(a, U2( 1, -1)) * (T)(M_SQRT1_2); }
  return U2(a.y + a.x, a.y - a.x) * M_SQRT1_2;
}

T2 mul_3t8(T2 a) { // mul(a, U2(-1, -1)) * (T)(M_SQRT1_2); }
  return U2(a.y - a.x, -a.y -a.x) * M_SQRT1_2;
}

T2 swap(T2 a)      { return U2(IM(a), RE(a)); }
T2 conjugate(T2 a) { return U2(RE(a), -IM(a)); }

T2 addsub(T2 a) { return U2(RE(a) + IM(a), RE(a) - IM(a)); }

// Same as X2(a, b), b = mul_t4(b)
#define X2_mul_t4(a, b) { T2 t = a; a = t + b; t.x = RE(b) - t.x; RE(b) = t.y - IM(b); IM(b) = t.x; }

#define X2(a, b) { T2 t = a; a = t + b; b = t - b; }

// Same as X2(a, conjugate(b))
#define X2conjb(a, b) { T2 t = a; RE(a) = RE(a) + RE(b); IM(a) = IM(a) - IM(b); RE(b) = t.x - RE(b); IM(b) = t.y + IM(b); }

// Same as X2(a, b), a = conjugate(a)
#define X2conja(a, b) { T2 t = a; RE(a) = RE(a) + RE(b); IM(a) = -IM(a) - IM(b); b = t - b; }

#define SWAP(a, b) { T2 t = a; a = b; b = t; }

T2 fmaT2(T a, T2 b, T2 c) { return a * b + c; }

// Partial complex multiplies:  the mul by sin is delayed so that it can be later propagated to an FMA instruction
// complex mul by cos-i*sin given cos/sin, sin
T2 partial_cmul(T2 a, T c_over_s) { return U2(mad(RE(a), c_over_s, IM(a)), mad(IM(a), c_over_s, -RE(a))); }
// complex mul by cos+i*sin given cos/sin, sin
T2 partial_cmul_conjugate(T2 a, T c_over_s) { return U2(mad(RE(a), c_over_s, -IM(a)), mad(IM(a), c_over_s, RE(a))); }

// a = c + sin * d; b = c - sin * d;
#define fma_addsub(a, b, sin, c, d) { d = sin * d; T2 t = c + d; b = c - d; a = t; }

// a * conjugate(b)
// saves one negation
T2 mul_by_conjugate(T2 a, T2 b) { return U2(RE(a) * RE(b) + IM(a) * IM(b), IM(a) * RE(b) - RE(a) * IM(b)); }

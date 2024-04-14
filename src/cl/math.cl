// Copyright (C) Mihai Preda

T2 U2(T a, T b) { return (T2) (a, b); }

double mad1(double x, double y, double z) { return x * y + z; }
// fma(x, y, z); }

T add1_m2(T x, T y) { return 2 * (x + y); }
T sub1_m2(T x, T y) { return 2 * (x - y); }

// x * y * 2
T mul1_m2(T x, T y) { return x * y * 2; }


OVERLOAD T fancyMul(T x, const T y) {
  // x * (y + 1);
  return fma(x, y, x);
}

OVERLOAD T2 fancyMul(T2 x, const T2 y) {
  return U2(fancyMul(RE(x), RE(y)), fancyMul(IM(x), IM(y)));
}

// complex square
OVERLOAD T2 sq(T2 a) { return U2(mad1(RE(a), RE(a), - IM(a) * IM(a)), mul1_m2(RE(a), IM(a))); }

// complex mul
OVERLOAD T2 mul(T2 a, T2 b) { return U2(mad1(RE(a), RE(b), -IM(a)*IM(b)), mad1(RE(a), IM(b), IM(a)*RE(b))); }

// complex add * 2
T2 add_m2(T2 a, T2 b) { return U2(add1_m2(RE(a), RE(b)), add1_m2(IM(a), IM(b))); }

// complex mul * 2
T2 mul_m2(T2 a, T2 b) { return 2 * U2(mad1(RE(a), RE(b), -IM(a)*IM(b)), mad1(RE(a), IM(b), IM(a)*RE(b))); }

// complex mul * 4
T2 mul_m4(T2 a, T2 b) { return 4 * U2(mad1(RE(a), RE(b), -IM(a)*IM(b)), mad1(RE(a), IM(b), IM(a)*RE(b))); }

// complex fma
T2 mad_m1(T2 a, T2 b, T2 c) { return U2(mad1(RE(a), RE(b), mad1(IM(a), -IM(b), RE(c))), mad1(RE(a), IM(b), mad1(IM(a), RE(b), IM(c)))); }

T2 mul_t4(T2 a)  { return U2(IM(a), -RE(a)); } // mul(a, U2( 0, -1)); }


T2 mul_t8(T2 a)  { return U2(IM(a) + RE(a), IM(a) - RE(a)) *   M_SQRT1_2; }  // mul(a, U2( 1, -1)) * (T)(M_SQRT1_2); }
T2 mul_3t8(T2 a) { return U2(RE(a) - IM(a), RE(a) + IM(a)) * - M_SQRT1_2; }  // mul(a, U2(-1, -1)) * (T)(M_SQRT1_2); }

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
T2 partial_cmul(T2 a, T c_over_s) { return U2(mad1(RE(a), c_over_s, IM(a)), mad1(IM(a), c_over_s, -RE(a))); }
// complex mul by cos+i*sin given cos/sin, sin
T2 partial_cmul_conjugate(T2 a, T c_over_s) { return U2(mad1(RE(a), c_over_s, -IM(a)), mad1(IM(a), c_over_s, RE(a))); }

// a = c + sin * d; b = c - sin * d;
#define fma_addsub(a, b, sin, c, d) { d = sin * d; T2 t = c + d; b = c - d; a = t; }

// a * conjugate(b)
// saves one negation
T2 mul_by_conjugate(T2 a, T2 b) { return U2(RE(a) * RE(b) + IM(a) * IM(b), IM(a) * RE(b) - RE(a) * IM(b)); }

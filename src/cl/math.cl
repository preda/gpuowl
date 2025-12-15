// Copyright (C) Mihai Preda

#pragma once

#include "base.cl"

// Access parts of a 64-bit value

u32 OVERLOAD lo32(u64 x) { uint2 x2 = as_uint2(x); return (u32)x2.x; }
u32 OVERLOAD hi32(u64 x) { uint2 x2 = as_uint2(x); return (u32)x2.y; }
u32 OVERLOAD lo32(i64 x) { uint2 x2 = as_uint2(x); return (u32)x2.x; }
i32 OVERLOAD hi32(i64 x) { uint2 x2 = as_uint2(x); return (i32)x2.y; }

// A primitive partial implementation of an i96 integer type
#if 1
// An all 32-bit implementation.  The add and subtract routines desperately need to use ASM with add.cc and sub.cc PTX instructions.
// This version might be best on AMD and Intel if we can generate add-with-carry instructions.
typedef struct { i32 hi32; u32 mid32; u32 lo32; } i96;
i96 OVERLOAD make_i96(i64 v) { i96 val; val.lo32 = lo32(v); val.mid32 = hi32(v); val.hi32 = (i32)val.mid32 >> 31; return val; }
i96 OVERLOAD make_i96(i32 v) { i96 val; val.lo32 = v; val.mid32 = val.hi32 = (i32)val.lo32 >> 31; return val; }
i96 OVERLOAD make_i96(i64 hi, u64 lo) { i96 val; val.hi32 = hi; val.mid32 = hi32(lo); val.lo32 = lo32(lo); return val; }
i96 OVERLOAD make_i96(i32 hi, u64 lo) { i96 val; val.hi32 = hi; val.mid32 = hi32(lo); val.lo32 = lo32(lo); return val; }
u32 i96_hi32(i96 val) { return val.hi32; }
u32 i96_mid32(i96 val) { return val.mid32; }
u32 i96_lo32(i96 val) { return val.lo32; }
u64 i96_lo64(i96 val) { return as_ulong((uint2)(val.lo32, val.mid32)); }
u64 i96_hi64(i96 val) { return as_ulong((uint2)(val.mid32, val.hi32)); }
i96 OVERLOAD add(i96 a, i96 b) {
  i96 val;
#if HAS_PTX >= 200        // Carry instructions requires sm_20 support or higher
  __asm("add.cc.u32  %0, %3, %6;\n\t"
        "addc.cc.u32 %1, %4, %7;\n\t"
        "addc.u32    %2, %5, %8;"
        : "=r"(val.lo32), "=r"(val.mid32), "=r"(val.hi32)
        : "r"(a.lo32), "r"(a.mid32), "r"(a.hi32), "r"(b.lo32), "r"(b.mid32), "r"(b.hi32));
#else
  u64 alo64 = as_ulong((uint2)(a.lo32, a.mid32)); u64 blo64 = as_ulong((uint2)(b.lo32, b.mid32)); u64 lo64 = alo64 + blo64;
  val.lo32 = lo32(lo64); val.mid32 = hi32(lo64); val.hi32 = a.hi32 + b.hi32 + (lo64 < alo64);
#endif
  return val;
}
i96 OVERLOAD add(i96 a, i64 b) { return add(a, make_i96(b)); }
i96 OVERLOAD sub(i96 a, i96 b) {
  i96 val;
#if HAS_PTX >= 200        // Carry instructions requires sm_20 support or higher
  __asm("sub.cc.u32  %0, %3, %6;\n\t"
        "subc.cc.u32 %1, %4, %7;\n\t"
        "subc.u32    %2, %5, %8;"
        : "=r"(val.lo32), "=r"(val.mid32), "=r"(val.hi32)
        : "r"(a.lo32), "r"(a.mid32), "r"(a.hi32), "r"(b.lo32), "r"(b.mid32), "r"(b.hi32));
#else
  u64 alo64 = as_ulong((uint2)(a.lo32, a.mid32)); u64 blo64 = as_ulong((uint2)(b.lo32, b.mid32)); u64 lo64 = alo64 - blo64;
  val.lo32 = lo32(lo64); val.mid32 = hi32(lo64); val.hi32 = a.hi32 - b.hi32 - (lo64 > alo64);
#endif
  return val;
}
i96 OVERLOAD sub(i96 a, i64 b) { return sub(a, make_i96(b)); }
i96 OVERLOAD sub(i96 a, i32 b) { return sub(a, make_i96(b)); }
#elif defined(__SIZEOF_INT128__)
// An i128 implementation.  This might use more GPU registers.  nVidia likes this version.
typedef struct { __int128 x; } i96;
i96 OVERLOAD make_i96(i64 v) { i96 val; val.x = v; return val; }
i96 OVERLOAD make_i96(i32 v) { i96 val; val.x = v; return val; }
i96 OVERLOAD make_i96(i64 hi, u64 lo) { i96 val; val.x = ((unsigned __int128)hi << 64) + lo; return val; }
i96 OVERLOAD make_i96(i32 hi, u64 lo) { return make_i96((i64)hi, lo); }
u32 i96_hi32(i96 val) { return (unsigned __int128)val.x >> 64; }
u32 i96_mid32(i96 val) { return (u64)val.x >> 32; }
u32 i96_lo32(i96 val) { return val.x; }
u64 i96_lo64(i96 val) { return val.x; }
u64 i96_hi64(i96 val) { return (unsigned __int128)val.x >> 32; }
i96 OVERLOAD add(i96 a, i96 b) { i96 val; val.x = a.x + b.x; return val; }
i96 OVERLOAD add(i96 a, i64 b) { return add(a, make_i96(b)); }
i96 OVERLOAD sub(i96 a, i96 b) { i96 val; val.x = a.x - b.x; return val; }
i96 OVERLOAD sub(i96 a, i64 b) { return sub(a, make_i96(b)); }
i96 OVERLOAD sub(i96 a, i32 b) { return sub(a, make_i96(b)); }
#elif 1
// A u64 lo32, u32 hi32 implementation.  This too would benefit from add with carry instructions.
// On nVidia, the clang optimizer kept the hi32 value as 64-bits!
typedef struct { u64 lo64; u32 hi32; } i96;
i96 OVERLOAD make_i96(i64 v) { i96 val; val.hi32 = v >> 63, val.lo64 = v; return val; }
i96 OVERLOAD make_i96(i32 v) { return make_i96((i64)v); }
i96 OVERLOAD make_i96(i64 hi, u64 lo) { i96 val; val.hi32 = hi, val.lo64 = lo; return val; }
i96 OVERLOAD make_i96(i32 hi, u64 lo) { i96 val; val.hi32 = hi, val.lo64 = lo; return val; }
u32 i96_hi32(i96 val) { return val.hi32; }
u32 i96_mid32(i96 val) { return hi32(val.lo64); }
u32 i96_lo32(i96 val) { return val.lo64; }
u64 i96_lo64(i96 val) { return val.lo64; }
u64 i96_hi64(i96 val) { return ((u64) val.hi32 << 32) | i96_mid32(val); }
i96 OVERLOAD add(i96 a, i96 b) { i96 val; val.lo64 = a.lo64 + b.lo64; val.hi32 = a.hi32 + b.hi32 + (val.lo64 < a.lo64); return val; }
i96 OVERLOAD add(i96 a, i64 b) { return add(a, make_i96(b)); }
i96 OVERLOAD sub(i96 a, i96 b) { i96 val; val.lo64 = a.lo64 - b.lo64; val.hi32 = a.hi32 - b.hi32 - (val.lo64 > a.lo64); return val; }
i96 OVERLOAD sub(i96 a, i64 b) { return sub(a, make_i96(b)); }
i96 OVERLOAD sub(i96 a, i32 b) { return sub(a, make_i96(b)); }
#endif

// A primitive partial implementation of an i128 and u128 integer type
#if defined(__SIZEOF_INT128__)
typedef struct { __int128 x; } i128;
typedef struct { unsigned __int128 x; } u128;
i128 OVERLOAD make_i128(i64 hi, u64 lo) { i128 val; val.x = ((__int128)hi << 64) | lo; return val; }
i128 OVERLOAD make_i128(u64 hi, u64 lo) { i128 val; val.x = ((__int128)hi << 64) | lo; return val; }
u64 i128_lo64(i128 val) { return val.x; }
u64 i128_shrlo64(i128 val, u32 bits) { return val.x >> bits; }
i128 OVERLOAD i128_masklo64(i128 a, u64 m) { i128 val; val.x = a.x & (((__int128)0xFFFFFFFFFFFFFFFFULL << 64) | m); return val; }
i128 OVERLOAD add(i128 a, i128 b) { i128 val; val.x = a.x + b.x; return val; }
i128 OVERLOAD add(i128 a, u64 b) { i128 val; val.x = a.x + (__int128)b; return val; }
i128 OVERLOAD add(i128 a, i64 b) { i128 val; val.x = a.x + (__int128)b; return val; }
i128 OVERLOAD sub(i128 a, u64 b) { i128 val; val.x = a.x - (__int128)b; return val; }
i128 OVERLOAD sub(i128 a, i64 b) { i128 val; val.x = a.x - (__int128)b; return val; }
u128 OVERLOAD make_u128(u64 hi, u64 lo) { u128 val; val.x = ((unsigned __int128)hi << 64) | lo; return val; }
u64 u128_lo64(u128 val) { return val.x; }
u64 u128_hi64(u128 val) { return val.x >> 64; }
u128 mul64(u64 a, u64 b) { u128 val; val.x = (unsigned __int128)a * (unsigned __int128)b; return val; }
u128 OVERLOAD add(u128 a, u128 b) { u128 val; val.x = a.x + b.x; return val; }
#else           // UNTESTED!  The mul64 macro causes clang to hang!
typedef struct { i64 hi64; u64 lo64; } i128;
typedef struct { u64 hi64; u64 lo64; } u128;
i128 OVERLOAD make_i128(i64 hi, u64 lo) { i128 val; val.hi64 = hi; val.lo64 = lo; return val; }
i128 OVERLOAD make_i128(u64 hi, u64 lo) { i128 val; val.hi64 = hi; val.lo64 = lo; return val; }
u64 i128_lo64(i128 val) { return val.lo64; }
u64 i128_shrlo64(i128 val, u32 bits) { return (val.hi64 << (64 - bits)) | (val.lo64 >> bits); }
i128 OVERLOAD i128_masklo64(i128 a, u64 m) { i128 val; val.lo64 = a.lo64 & m; val.hi64 = a.hi64; return val; }
i128 OVERLOAD add(i128 a, i128 b) { i128 val; val.lo64 = a.lo64 + b.lo64; val.hi64 = a.hi64 + b.hi64 + (val.lo64 < a.lo64); return val; }
i128 OVERLOAD add(i128 a, u64 b) { i128 val; val.lo64 = a.lo64 + b; val.hi64 = a.hi64 + (val.lo64 < a.lo64); return val; }
i128 OVERLOAD add(i128 a, i64 b) { i128 val; val.lo64 = a.lo64 + b; val.hi64 = a.hi64 + (b >> 63) + (val.lo64 < a.lo64); return val; }
i128 OVERLOAD sub(i128 a, u64 b) { i128 val; val.lo64 = a.lo64 - b; val.hi64 = a.hi64 - (val.lo64 > a.lo64); return val; }
i128 OVERLOAD sub(i128 a, i64 b) { i128 val; val.lo64 = a.lo64 - (u64)b; val.hi64 = a.hi64 - (b >> 63) - (val.lo64 > a.lo64); return val; }
u128 OVERLOAD make_u128(u64 hi, u64 lo) { u128 val; val.hi64 = hi; val.lo64 = lo; return val; }
u64 u128_lo64(u128 val) { return val.lo64; }
u64 u128_hi64(u128 val) { return val.hi64; }
u128 mul64(u64 a, u64 b) { u128 val; val.lo64 = a * b; val.hi64 = mul_hi(a, b); return val; }
u128 OVERLOAD add(u128 a, u128 b) { u128 val; val.lo64 = a.lo64 + b.lo64; val.hi64 = a.hi64 + b.hi64 + (val.lo64 < a.lo64); return val; }
#endif

// Select based on sign of first argument.  This generates less PTX code, but is no faster on 5xxx GPUs
i32 select32(i32 a, i32 b, i32 c) {
#if HAS_PTX >= 100        // slct instruction requires sm_10 support or higher
  i32 res;
  __asm("slct.s32.s32 %0, %2, %3, %1;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  return a >= 0 ? b : c;
#endif
}

// Optionally add a value if first arg is negative.
i32 optional_add(i32 a, const i32 b) {
#if HAS_PTX >= 100        // setp/add instruction requires sm_10 support or higher
  __asm("{.reg .pred %%p;\n\t"
        " setp.lt.s32 %%p, %0, 0;\n\t"    // a < 0
        " @%%p add.s32 %0, %0, %1;}"      // if (a < 0) a = a + b
        : "+r"(a) : "n"(b));
#else
  if (a < 0) a = a + b;
#endif
  return a;
}

// Optionally subtract a value if first arg is negative.
i32 optional_sub(i32 a, const i32 b) {
#if HAS_PTX >= 100        // setp/sub instruction requires sm_10 support or higher
  __asm("{.reg .pred %%p;\n\t"
        " setp.lt.s32 %%p, %0, 0;\n\t"    // a < 0
        " @%%p sub.s32 %0, %0, %1;}"      // if (a < 0) a = a - b
        : "+r"(a) : "n"(b));
#else
  if (a < 0) a = a - b;
#endif
  return a;
}

// Optionally subtract a value if first arg is greater than value.
i32 optional_mod(i32 a, const i32 b) {
#if 0 //HAS_PTX >= 100        // setp/sub instruction requires sm_10 support or higher  // Not faster on 5xxx GPUs (not sure why)
  __asm("{.reg .pred %%p;\n\t"
        " setp.ge.s32 %%p, %0, %1;\n\t"   // a > b
        " @%%p sub.s32 %0, %0, %1;}"      // if (a > b) a = a - b
        : "+r"(a) : "n"(b));
#else
  if (a >= b) a = a - b;
#endif
  return a;
}

// Multiply and add primitives

u64 OVERLOAD mad32(u32 a, u32 b, u32 c) {
#if HAS_PTX >= 200        // mad instruction requires sm_20 support or higher           // Same speed on TitanV, any gain may be too small to measure
  u32 reslo, reshi;
  __asm("mad.lo.cc.u32 %0, %2, %3, %4;\n\t"
        "madc.hi.u32   %1, %2, %3, 0;" : "=r"(reslo), "=r"(reshi) : "r"(a), "r"(b), "r"(c));
  return as_ulong((uint2)(reslo, reshi));
#else
  return (u64)a * (u64)b + c;
#endif
}

u64 OVERLOAD mad32(u32 a, u32 b, u64 c) {
#if HAS_PTX >= 200        // mad instruction requires sm_20 support or higher          // Same speed on TitanV, any gain may be too small to measure
  u32 reslo, reshi;
  __asm("mad.lo.cc.u32 %0, %2, %3, %4;\n\t"
        "madc.hi.u32   %1, %2, %3, %5;" : "=r"(reslo), "=r"(reshi) : "r"(a), "r"(b), "r"(lo32(c)), "r"(hi32(c)));
  return as_ulong((uint2)(reslo, reshi));
#else
  return (u64)a * (u64)b + c;
#endif
}

u128 OVERLOAD mad64(u64 a, u64 b, u64 c) {
#if 0 && HAS_PTX >= 200        // mad instruction requires sm_20 support or higher    // Slower on TitanV and mobile 4070, don't understand why
  u64 reslo, reshi;
  __asm("mad.lo.cc.u64 %0, %2, %3, %4;\n\t"
        "madc.hi.u64   %1, %2, %3, 0;" : "=l"(reslo), "=l"(reshi) : "l"(a), "l"(b), "l"(u128_lo64(c)));
  return make_u128(reshi, reslo);
#elif HAS_PTX >= 200        // mad instruction requires sm_20 support or higher       // Faster on TitanV.  No difference on mobile 4070.  Much cleaner PTX code generated.
  uint2 a2 = as_uint2(a);
  uint2 b2 = as_uint2(b);
  uint2 c2 = as_uint2(c);
  uint2 rlo2, rhi2;
  __asm("mad.lo.cc.u32  %0, %4, %6, %8;\n\t"
        "madc.hi.cc.u32 %1, %4, %6, %9;\n\t"
        "madc.lo.cc.u32 %2, %5, %7, 0;\n\t"
        "madc.hi.u32    %3, %5, %7, 0;\n\t"
        "mad.lo.cc.u32  %1, %5, %6, %1;\n\t"
        "madc.hi.cc.u32 %2, %5, %6, %2;\n\t"
        "addc.u32       %3, %3, 0;\n\t"
        "mad.lo.cc.u32  %1, %4, %7, %1;\n\t"
        "madc.hi.cc.u32 %2, %4, %7, %2;\n\t"
        "addc.u32       %3, %3, 0;"
        : "=r"(rlo2.x), "=r"(rlo2.y), "=r"(rhi2.x), "=r"(rhi2.y)
        : "r"(a2.x), "r"(a2.y), "r"(b2.x), "r"(b2.y), "r"(c2.x), "r"(c2.y));
  return make_u128((u64)as_ulong(rhi2), (u64)as_ulong(rlo2));
#else
  return add(mul64(a, b), make_u128(0, c));
#endif
}

u128 OVERLOAD mad64(u64 a, u64 b, u128 c) {
#if 0 && HAS_PTX >= 200        // mad instruction requires sm_20 support or higher  // Slower on TitanV and mobile 4070, don't understand why
  u64 reslo, reshi;
  __asm("mad.lo.cc.u64 %0, %2, %3, %4;\n\t"
        "madc.hi.u64   %1, %2, %3, %5;" : "=l"(reslo), "=l"(reshi) : "l"(a), "l"(b), "l"(u128_lo64(c)), "l"(u128_hi64(c)));
  return make_u128(reshi, reslo);
#elif HAS_PTX >= 200        // mad instruction requires sm_20 support or higher     // Faster on TitanV.  No difference on mobile 4070.  Much cleaner PTX code generated.
  uint2 a2 = as_uint2(a);
  uint2 b2 = as_uint2(b);
  uint2 clo2 = as_uint2(u128_lo64(c));
  uint2 chi2 = as_uint2(u128_hi64(c));
  uint2 rlo2, rhi2;
  __asm("mad.lo.cc.u32  %0, %4, %6, %8;\n\t"
        "madc.hi.cc.u32 %1, %4, %6, %9;\n\t"
        "madc.lo.cc.u32 %2, %5, %7, %10;\n\t"
        "madc.hi.u32    %3, %5, %7, %11;\n\t"
        "mad.lo.cc.u32  %1, %5, %6, %1;\n\t"
        "madc.hi.cc.u32 %2, %5, %6, %2;\n\t"
        "addc.u32       %3, %3, 0;\n\t"
        "mad.lo.cc.u32  %1, %4, %7, %1;\n\t"
        "madc.hi.cc.u32 %2, %4, %7, %2;\n\t"
        "addc.u32       %3, %3, 0;"
        : "=r"(rlo2.x), "=r"(rlo2.y), "=r"(rhi2.x), "=r"(rhi2.y)
        : "r"(a2.x), "r"(a2.y), "r"(b2.x), "r"(b2.y), "r"(clo2.x), "r"(clo2.y), "r"(chi2.x), "r"(chi2.y));
  return make_u128((u64)as_ulong(rhi2), (u64)as_ulong(rlo2));
#else
  return add(mul64(a, b), c);
#endif
}


// The X2 family of macros and SWAP are #defines because OpenCL does not allow pass by reference.
// With NTT support added, we need to turn these macros into overloaded routines.
#define X2(a, b)              X2_internal(&(a), &(b))                 // a = a + b, b = a - b
#define X2conjb(a, b)         X2conjb_internal(&(a), &(b))            // X2(a, conjugate(b))
#define X2_mul_t4(a, b)       X2_mul_t4_internal(&(a), &(b))          // X2(a, b), b = mul_t4(b)
#define X2_mul_t8(a, b)       X2_mul_t8_internal(&(a), &(b))          // X2(a, b), b = mul_t8(b)
#define X2_mul_3t8(a, b)      X2_mul_3t8_internal(&(a), &(b))         // X2(a, b), b = mul_3t8(b)
#define X2_conjb(a, b)        X2_conjb_internal(&(a), &(b))           // X2(a, b), b = conjugate(b)
#define SWAP(a, b)            SWAP_internal(&(a), &(b))               // a = b, b = a
#define SWAP_XY(a)            U2((a).y, (a).x)                        // Swap real and imaginary components of a

#if FFT_FP64

T2 OVERLOAD conjugate(T2 a) { return U2(a.x, -a.y); }

// Multiply by 2 without using floating point instructions.  This is a little sloppy as an input of zero returns 2^-1022.
T OVERLOAD mul2(T a) { int2 tmp = as_int2(a); tmp.y += 0x00100000; /* Bump exponent by 1 */ return (as_double(tmp)); }
T2 OVERLOAD mul2(T2 a) { return U2(mul2(a.x), mul2(a.y)); }

// Multiply by -2 without using floating point instructions.  This is a little sloppy as an input of zero returns -2^-1022.
T OVERLOAD mulminus2(T a) { int2 tmp = as_int2(a); tmp.y += 0x80100000; /* Bump exponent by 1, flip sign bit */ return (as_double(tmp)); }
T2 OVERLOAD mulminus2(T2 a) { return U2(mulminus2(a.x), mulminus2(a.y)); }

// a * (b + 1) == a * b + a
T OVERLOAD fancyMul(T a, T b)   { return fma(a, b, a); }
T2 OVERLOAD fancyMul(T2 a, T2 b) { return U2(fancyMul(a.x, b.x), fancyMul(a.y, b.y)); }

// Square a complex number
T2 OVERLOAD csq(T2 a) { return U2(fma(a.x, a.x, - a.y * a.y), mul2(a.x) * a.y); }
// a^2 + c
T2 OVERLOAD csqa(T2 a, T2 c) { return U2(fma(a.x, a.x, fma(a.y, -a.y, c.x)), fma(mul2(a.x), a.y, c.y)); }
// Same as csq(a), -a
T2 OVERLOAD csq_neg(T2 a) { return U2(fma(-a.x, a.x, a.y * a.y), mulminus2(a.x) * a.y); }                               // NOT USED

// Complex multiply
T2 OVERLOAD cmul(T2 a, T2 b) { return U2(fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x)); }

T2 OVERLOAD cfma(T2 a, T2 b, T2 c) { return U2(fma(a.x, b.x, fma(a.y, -b.y, c.x)), fma(a.y, b.x, fma(a.x, b.y, c.y))); }

T2 OVERLOAD cmul_by_conjugate(T2 a, T2 b) { return cmul(a, conjugate(b)); }

// Multiply a by b and conjugate(b).  This saves 2 multiplies.
void OVERLOAD cmul_a_by_b_and_conjb(T2 *res1, T2 *res2, T2 a, T2 b) {
  T axbx = a.x * b.x;
  T aybx = a.y * b.x;
  res1->x = fma(a.y, -b.y, axbx), res1->y = fma(a.x,  b.y, aybx);
  res2->x = fma(a.y,  b.y, axbx), res2->y = fma(a.x, -b.y, aybx);
}

// Square a (cos,sin) complex number.  Fancy squaring returns a fancy value.  Defancy squares a fancy number returning a non-fancy number.
T2 OVERLOAD csqTrig(T2 a) { T two_ay = mul2(a.y); return U2(fma(-two_ay, a.y, 1), a.x * two_ay); }
T2 csqTrigFancy(T2 a) { T two_ay = mul2(a.y); return U2(-two_ay * a.y, fma(a.x, two_ay, two_ay)); }
T2 csqTrigDefancy(T2 a) { T two_ay = mul2(a.y); return U2(fma (-two_ay, a.y, 1), fma(a.x, two_ay, two_ay)); }

// Cube a complex number w (cos,sin) given w^2 and w.  The squared input can be either fancy or not fancy.
// Fancy cCube takes a fancy w argument and returns a fancy value.  Defancy takes a fancy w argument and returns a non-fancy value.
T2 OVERLOAD ccubeTrig(T2 sq, T2 w) { T tmp = mul2(sq.y); return U2(fma(tmp, -w.y, w.x), fma(tmp, w.x, -w.y)); }
T2 ccubeTrigFancy(T2 sq, T2 w) { T tmp = mul2(sq.y); return U2(fma(tmp, -w.y, w.x), fma(tmp, w.x, tmp - w.y)); }
T2 ccubeTrigDefancy(T2 sq, T2 w) { T tmp = mul2(sq.y); T wx = w.x + 1; return U2(fma(tmp, -w.y, wx), fma(tmp, wx, -w.y)); }

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

T2 OVERLOAD mul_t4(T2 a)  { return U2(-a.y, a.x); } // i.e. a * i

T2 OVERLOAD mul_t8(T2 a)  { // mul(a, U2(1, 1)) * (T)(M_SQRT1_2); }
  // One mul, two FMAs
  T ay = a.y * M_SQRT1_2;
  return U2(fma(a.x, M_SQRT1_2, -ay), fma(a.x, M_SQRT1_2, ay));
// Two adds, two muls
//  return U2(a.x - a.y, a.x + a.y) * M_SQRT1_2;
}

T2 OVERLOAD mul_3t8(T2 a) { // mul(a, U2(-1, 1)) * (T)(M_SQRT1_2); }
  // One mul, two FMAs
  T ay = a.y * M_SQRT1_2;
  return U2(fma(-a.x, M_SQRT1_2, -ay), fma(a.x, M_SQRT1_2, -ay));
// Two adds, two muls
//  return U2(-(a.x + a.y), a.x - a.y) * M_SQRT1_2;
}

// Return a+b and a-b
void OVERLOAD X2_internal(T2 *a, T2 *b) { T2 t = *a; *a = t + *b; *b = t - *b; }

// Same as X2(a, b), b = mul_t4(b)
void OVERLOAD X2_mul_t4_internal(T2 *a, T2 *b) { T2 t = *a; *a = *a + *b; t.x = t.x - b->x; b->x = b->y - t.y; b->y = t.x; }

// Same as X2(a, conjugate(b))
void OVERLOAD X2conjb_internal(T2 *a, T2 *b) { T2 t = *a; a->x = a->x + b->x; a->y = a->y - b->y; b->x = t.x - b->x; b->y = t.y + b->y; }

// Same as X2(a, b), b = conjugate(b)
void OVERLOAD X2_conjb_internal(T2 *a, T2 *b) { T2 t = *a; *a = t + *b; b->x = t.x - b->x; b->y = b->y - t.y; }

void OVERLOAD SWAP_internal(T2 *a, T2 *b) { T2 t = *a; *a = *b; *b = t; }

T2 fmaT2(T a, T2 b, T2 c) { return fma(U2(a, a), b, c); }

// a = c + sin * d; b = c - sin * d;
#define fma_addsub(a, b, sin, c, d) { T2 t = c + sin * d; b = c - sin * d; a = t; }

T2 OVERLOAD addsub(T2 a) { return U2(a.x + a.y, a.x - a.y); }

// computes 2*(a.x*b.x+a.y*b.y) + i*2*(a.x*b.y+a.y*b.x)
// which happens to be the cyclical convolution (a.x, a.y)x(b.x, b.y) * 2
T2 OVERLOAD foo2(T2 a, T2 b) { a = addsub(a); b = addsub(b); return addsub(U2(RE(a) * RE(b), IM(a) * IM(b))); }

// computes 2*[x^2+y^2 + i*(2*x*y)]. i.e. 2 * cyclical autoconvolution of (x, y)
T2 OVERLOAD foo(T2 a) { return foo2(a, a); }

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

F2 OVERLOAD conjugate(F2 a) { return U2(a.x, -a.y); }

// Multiply by 2 without using floating point instructions.  This is a little sloppy as an input of zero returns 2^-126.
F OVERLOAD mul2(F a) { return a + a; }  //{ int tmp = as_int(a); tmp += 0x00800000; /* Bump exponent by 1 */ return (as_float(tmp)); }
F2 OVERLOAD mul2(F2 a) { return U2(mul2(a.x), mul2(a.y)); }

// Multiply by -2 without using floating point instructions.  This is a little sloppy as an input of zero returns -2^-126.
F OVERLOAD mulminus2(F a) { return -2.0f * a; } //{ int tmp = as_int(a); tmp += 0x80800000; /* Bump exponent by 1, flip sign bit */ return (as_float(tmp)); }
F2 OVERLOAD mulminus2(F2 a) { return U2(mulminus2(a.x), mulminus2(a.y)); }

// a * (b + 1) == a * b + a
F OVERLOAD fancyMul(F a, F b)   { return fma(a, b, a); }
F2 OVERLOAD fancyMul(F2 a, F2 b) { return U2(fancyMul(a.x, b.x), fancyMul(a.y, b.y)); }

// Square a complex number
F2 OVERLOAD csq(F2 a) { return U2(fma(a.x, a.x, - a.y * a.y), mul2(a.x) * a.y); }
// a^2 + c
F2 OVERLOAD csqa(F2 a, F2 c) { return U2(fma(a.x, a.x, fma(a.y, -a.y, c.x)), fma(mul2(a.x), a.y, c.y)); }
// Same as csq(a), -a
F2 OVERLOAD csq_neg(F2 a) { return U2(fma(-a.x, a.x, a.y * a.y), mulminus2(a.x) * a.y); }                               // NOT USED

// Complex multiply
F2 OVERLOAD cmul(F2 a, F2 b) { return U2(fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x)); }

F2 OVERLOAD cfma(F2 a, F2 b, F2 c) { return U2(fma(a.x, b.x, fma(a.y, -b.y, c.x)), fma(a.y, b.x, fma(a.x, b.y, c.y))); }

F2 OVERLOAD cmul_by_conjugate(F2 a, F2 b) { return cmul(a, conjugate(b)); }

// Multiply a by b and conjugate(b).  This saves 2 multiplies.
void OVERLOAD cmul_a_by_b_and_conjb(F2 *res1, F2 *res2, F2 a, F2 b) {
  F axbx = a.x * b.x;
  F aybx = a.y * b.x;
  res1->x = fma(a.y, -b.y, axbx), res1->y = fma(a.x,  b.y, aybx);
  res2->x = fma(a.y,  b.y, axbx), res2->y = fma(a.x, -b.y, aybx);
}

// Square a (cos,sin) complex number.  Fancy squaring returns a fancy value.  Defancy squares a fancy number returning a non-fancy number.
F2 OVERLOAD csqTrig(F2 a) { F two_ay = mul2(a.y); return U2(fma(-two_ay, a.y, 1), a.x * two_ay); }
F2 csqTrigFancy(F2 a) { F two_ay = mul2(a.y); return U2(-two_ay * a.y, fma(a.x, two_ay, two_ay)); }
F2 csqTrigDefancy(F2 a) { F two_ay = mul2(a.y); return U2(fma (-two_ay, a.y, 1), fma(a.x, two_ay, two_ay)); }

// Cube a complex number w (cos,sin) given w^2 and w.  The squared input can be either fancy or not fancy.
// Fancy cCube takes a fancy w argument and returns a fancy value.  Defancy takes a fancy w argument and returns a non-fancy value.
F2 OVERLOAD ccubeTrig(F2 sq, F2 w) { F tmp = mul2(sq.y); return U2(fma(tmp, -w.y, w.x), fma(tmp, w.x, -w.y)); }
F2 ccubeTrigFancy(F2 sq, F2 w) { F tmp = mul2(sq.y); return U2(fma(tmp, -w.y, w.x), fma(tmp, w.x, tmp - w.y)); }
F2 ccubeTrigDefancy(F2 sq, F2 w) { F tmp = mul2(sq.y); F wx = w.x + 1; return U2(fma(tmp, -w.y, wx), fma(tmp, wx, -w.y)); }

// Complex a * (b + 1)
// Useful for mul with twiddles of small angles, where the real part is stored with the -1 trick for increased precision
F2 cmulFancy(F2 a, F2 b) { return cfma(a, b, a); }

// Multiply a by fancy b and conjugate(fancy b).  This saves 2 FMAs.
void cmul_a_by_fancyb_and_conjfancyb(F2 *res1, F2 *res2, F2 a, F2 b) {
  F axbx = fma(a.x, b.x, a.x);
  F aybx = fma(a.y, b.x, a.y);
  res1->x = fma(a.y, -b.y, axbx), res1->y = fma(a.x,  b.y, aybx);
  res2->x = fma(a.y,  b.y, axbx), res2->y = fma(a.x, -b.y, aybx);
}

F2 OVERLOAD mul_t4(F2 a)  { return U2(-a.y, a.x); } // i.e. a * i

F2 OVERLOAD mul_t8(F2 a)  { // mul(a, U2(1, 1)) * (T)(M_SQRT1_2); }
  // One mul, two FMAs
  F ay = a.y * (float) M_SQRT1_2;
  return U2(fma(a.x, (float) M_SQRT1_2, -ay), fma(a.x, (float) M_SQRT1_2, ay));
// Two adds, two muls
//  return U2(a.x - a.y, a.x + a.y) * M_SQRT1_2;
}

F2 OVERLOAD mul_3t8(F2 a) { // mul(a, U2(-1, 1)) * (T)(M_SQRT1_2); }
  // One mul, two FMAs
  F ay = a.y * (float) M_SQRT1_2;
  return U2(fma(-a.x, (float) M_SQRT1_2, -ay), fma(a.x, (float) M_SQRT1_2, -ay));
// Two adds, two muls
//  return U2(-(a.x + a.y), a.x - a.y) * M_SQRT1_2;
}

// Return a+b and a-b
void OVERLOAD X2_internal(F2 *a, F2 *b) { F2 t = *a; *a = t + *b; *b = t - *b; }

// Same as X2(a, b), b = mul_t4(b)
void OVERLOAD X2_mul_t4_internal(F2 *a, F2 *b) { F2 t = *a; *a = *a + *b; t.x = t.x - b->x; b->x = b->y - t.y; b->y = t.x; }

// Same as X2(a, conjugate(b))
void OVERLOAD X2conjb_internal(F2 *a, F2 *b) { F2 t = *a; a->x = a->x + b->x; a->y = a->y - b->y; b->x = t.x - b->x; b->y = t.y + b->y; }

// Same as X2(a, b), b = conjugate(b)
void OVERLOAD X2_conjb_internal(F2 *a, F2 *b) { F2 t = *a; *a = t + *b; b->x = t.x - b->x; b->y = b->y - t.y; }

void OVERLOAD SWAP_internal(F2 *a, F2 *b) { F2 t = *a; *a = *b; *b = t; }

F2 fmaT2(F a, F2 b, F2 c) { return fma(U2(a, a), b, c); }

// a = c + sin * d; b = c - sin * d;
#define fma_addsub(a, b, sin, c, d) { F2 t = c + sin * d; b = c - sin * d; a = t; }

F2 OVERLOAD addsub(F2 a) { return U2(a.x + a.y, a.x - a.y); }

// computes 2*(a.x*b.x+a.y*b.y) + i*2*(a.x*b.y+a.y*b.x)
// which happens to be the cyclical convolution (a.x, a.y)x(b.x, b.y) * 2
F2 OVERLOAD foo2(F2 a, F2 b) { a = addsub(a); b = addsub(b); return addsub(U2(RE(a) * RE(b), IM(a) * IM(b))); }

// computes 2*[x^2+y^2 + i*(2*x*y)]. i.e. 2 * cyclical autoconvolution of (x, y)
F2 OVERLOAD foo(F2 a) { return foo2(a, a); }

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

// bits in reduced mod M.
#define M31 ((((Z31) 1) << 31) - 1)

#if 0          // Version that keeps results strictly in the 0..M31-1 range

Z31 OVERLOAD mod(Z31 a) { return (a & M31) + (a >> 31); }            // GWBUG: This could be larger than M31 (unless a is result of an add), need a wesk and strong mod

Z31 OVERLOAD add(Z31 a, Z31 b) { Z31 t = a + b; return t - (t >= M31 ? M31 : 0); }     //GWBUG - an if stmt may be faster
GF31 OVERLOAD add(GF31 a, GF31 b) { return U2(add(a.x, b.x), add(a.y, b.y)); }

Z31 OVERLOAD sub(Z31 a, Z31 b) { Z31 t = a - b; return t + (t >= M31 ? M31 : 0); }     //GWBUG - an if stmt may be faster.  So might "(i64) t < 0".
GF31 OVERLOAD sub(GF31 a, GF31 b) { return U2(sub(a.x, b.x), sub(a.y, b.y)); }

Z31 OVERLOAD neg(Z31 a) { return a == 0 ? 0 : M31 - a; }                // GWBUG: Examine all callers to see if neg call can be avoided
GF31 OVERLOAD neg(GF31 a) { return U2(neg(a.x), neg(a.y)); }

Z31 OVERLOAD make_Z31(i32 a) { return (Z31) (a < 0 ? a + M31 : a); }              // Handles signed values of a
Z31 OVERLOAD make_Z31(u32 a) { return (Z31) (a); }                                // a must be in range of 0 .. M31-1
Z31 OVERLOAD make_Z31(i64 a) { if (a < 0) a += (((i64) M31 << 31) + M31); return add((Z31) (a & M31), (Z31) (a >> 31)); } // Handles 62-bit a values

u32 get_Z31(Z31 a) { return a; }  // Get balanced value in range 0 to M31-1
i32 get_balanced_Z31(Z31 a) { return (a & 0xC0000000) ? (i32) a - M31 : (i32) a; }  // Get balanced value in range -M31/2 to M31/2

// Assumes k reduced mod 31.
Z31 OVERLOAD shl(Z31 a, u32 k) { return ((a << k) + (a >> (31 - k))) & M31; }
GF31 OVERLOAD shl(GF31 a, u32 k) { return U2(shl(a.x, k), shl(a.y, k)); }
Z31 OVERLOAD shr(Z31 a, u32 k) { return ((a >> k) + (a << (31 - k))) & M31; }
GF31 OVERLOAD shr(GF31 a, u32 k) { return U2(shr(a.x, k), shr(a.y, k)); }

Z31 OVERLOAD mul(Z31 a, Z31 b) { u64 t = a * (u64) b; return add((Z31) (t & M31), (Z31) (t >> 31)); }

// Multiply by 2
Z31 OVERLOAD mul2(Z31 a) { return add(a, a); }
GF31 OVERLOAD mul2(GF31 a) { return U2(mul2(a.x), mul2(a.y)); }

// Return conjugate of a
GF31 OVERLOAD conjugate(GF31 a) { return U2(a.x, neg(a.y)); }

// Complex square.  input, output 31 bits. Uses (a + i*b)^2 == ((a+b)*(a-b) + i*2*a*b).
GF31 OVERLOAD csq(GF31 a) { return U2(mul(add(a.x, a.y), sub(a.x, a.y)), mul2(mul(a.x, a.y))); }        //GWBUG: Probably faster to double a.y and have a mul that takes non-normalized inputs

// a^2 + c
GF31 OVERLOAD csqa(GF31 a, GF31 c) { return add(csq(a), c); }                                           // GWBUG: inline csq so we only "mod" after adding c??  Find a way to use fma instructions

// Complex mul
//GF31 OVERLOAD cmul(GF31 a, GF31 b) { return U2(sub(mul(a.x, b.x), mul(a.y, b.y)), add(mul(a.x, b.y), mul(a.y, b.x)));}   // GWBUG:  Is a 3 multiply complex mul faster?  See above
GF31 OVERLOAD cmul(GF31 a, GF31 b) {
  Z31 k1 = mul(b.x, add(a.x, a.y));
  Z31 k2 = mul(a.x, sub(b.y, b.x));
  Z31 k3 = mul(a.y, add(b.y, b.x));
  return U2(sub(k1, k3), add(k1, k2));
}

// mul with (0, 1). (twiddle of tau/4, sqrt(-1) aka "i").
GF31 OVERLOAD mul_t4(GF31 a) { return U2(neg(a.y), a.x); }                                              // GWBUG:  Can caller use a version that does not negate real?

// mul with (2^15, 2^15). (twiddle of tau/8 aka sqrt(i)). Note: 2 * (+/-2^15)^2 == 1 (mod M31).
GF31 OVERLOAD mul_t8(GF31 a) { return U2(shl(sub(a.x, a.y), 15), shl(add(a.x, a.y), 15)); }       // GWBUG:  Can caller use a version that does not negate real?  is shl(neg) same as shr???

// mul with (-2^15, 2^15). (twiddle of 3*tau/8).
GF31 OVERLOAD mul_3t8(GF31 a) { return U2(shl(neg(add(a.x, a.y)), 15), shl(sub(a.x, a.y), 15)); }

// Return a+b and a-b
void OVERLOAD X2_internal(GF31 *a, GF31 *b) { GF31 t = *a; *a = add(t, *b); *b = sub(t, *b); }

// Same as X2(a, conjugate(b))
void OVERLOAD X2conjb_internal(GF31 *a, GF31 *b) { GF31 t = *a; a->x = add(a->x, b->x); a->y = sub(a->y, b->y); b->x = sub(t.x, b->x); b->y = add(t.y, b->y); }

// Same as X2(a, b), b = mul_t4(b)
void OVERLOAD X2_mul_t4_internal(GF31 *a, GF31 *b) { GF31 t = *a; *a = add(*a, *b); t.x = sub(t.x, b->x); b->x = sub(b->y, t.y); b->y = t.x; }

// Same as X2(a, b), b = mul_t8(b)
void OVERLOAD X2_mul_t8_internal(GF31 *a, GF31 *b) { X2(*a, *b); *b = mul_t8(*b); }

// Same as X2(a, b), b = mul_3t8(b)
void OVERLOAD X2_mul_3t8_internal(GF31 *a, GF31 *b) { X2(*a, *b); *b = mul_3t8(*b); }           //GWBUG: can we do better (elim a negate)?

// Same as X2(a, b), b = conjugate(b)
void OVERLOAD X2_conjb_internal(GF31 *a, GF31 *b) { GF31 t = *a; *a = add(t, *b); b->x = sub(t.x, b->x); b->y = sub(b->y, t.y); }

void OVERLOAD SWAP_internal(GF31 *a, GF31 *b) { GF31 t = *a; *a = *b; *b = t; }

GF31 OVERLOAD addsub(GF31 a) { return U2(add(a.x, a.y), sub(a.x, a.y)); }
GF31 OVERLOAD foo2(GF31 a, GF31 b) { a = addsub(a); b = addsub(b); return addsub(U2(mul(RE(a), RE(b)), mul(IM(a), IM(b)))); }
GF31 OVERLOAD foo(GF31 a) { return foo2(a, a); }





#elif 1                                                               // This version is a little sloppy.  Returns values in 0..M31 range.

// Internal routines to return value in 0..M31 range
#if MODM31 == 0
Z31 OVERLOAD modM31(Z31 a) { return (a & M31) + (a >> 31); }                          // Assumes a is not 0xFFFFFFFF (which would return 0x80000000)
Z31 OVERLOAD modM31(i32 a) { return (a & M31) + (a >> 31); }                          // Assumes a is not 0x80000000 (which would return 0xFFFFFFFF)
#elif MODM31 == 1
Z31 OVERLOAD modM31(Z31 a) { i32 alt = a + 0x80000001; return select32(a, a, alt); }  // Assumes a is not 0xFFFFFFFF (which would return 0x80000000)
Z31 OVERLOAD modM31(i32 a) { i32 alt = a - 0x80000001; return select32(a, a, alt); }  // Assumes a is not 0x80000000 (which would return 0xFFFFFFFF)
#else
Z31 OVERLOAD modM31(Z31 a) { return optional_add(a, 0x80000001); }                    // Assumes a is not 0xFFFFFFFF (which would return 0x80000000)
Z31 OVERLOAD modM31(i32 a) { return optional_sub(a, 0x80000001); }                    // Assumes a is not 0x80000000 (which would return 0xFFFFFFFF)
#endif

Z31 OVERLOAD modM31(u64 a) {                                          // a must be less than 0xFFFFFFFF7FFFFFFF
  u32 alo = a & M31;
  u32 amid = (a >> 31) & M31;
  u32 ahi = a >> 62;
  return modM31(ahi + amid + alo);                                    // 32-bit overflow does not occur due to restrictions on input
}
Z31 OVERLOAD modM31(i64 a) {                                          // abs(a) must be less than 0x7FFFFFFF80000000
  u32 alo = a & M31;
  u32 amid = ((u64) a >> 31) & M31;                                   // Unsigned shift might be faster than signed shift
  u32 ahi = a >> 62;                                                  // Sign extend the top bits
  return modM31(ahi + amid + alo);                                    // This is where caller must assure a 32-bit overflow does not occur
}
Z31 OVERLOAD modM31q(u64 a) {                                         // Quick version, a < 2^62
  u32 alo = a & M31;
  u32 ahi = a >> 31;
  return modM31(ahi + alo);
}
#if 0                                           // GWBUG - which is faster?
Z31 OVERLOAD modM31q(i64 a) {                                         // Quick version, abs(a) must be 61 bits
  u32 alo = a & M31;
  i32 ahi = a >> 31;                                                  // Sign extend the top bits
  if (ahi < 0) ahi = ahi + M31;
  return modM31((u32) ahi + alo);
}
#else
Z31 OVERLOAD modM31q(i64 a) { return modM31(a); }                     // Quick version, abs(a) must be 61 bits
#endif

Z31 OVERLOAD neg(Z31 a) { return M31 - a; }                           // GWBUG: Examine all callers to see if neg call can be avoided
GF31 OVERLOAD neg(GF31 a) { return U2(neg(a.x), neg(a.y)); }

Z31 OVERLOAD add(Z31 a, Z31 b) { return modM31(a + b); }
GF31 OVERLOAD add(GF31 a, GF31 b) { return U2(add(a.x, b.x), add(a.y, b.y)); }

Z31 OVERLOAD sub(Z31 a, Z31 b) { i32 t = a - b; return (t & M31) + (t >> 31); }
GF31 OVERLOAD sub(GF31 a, GF31 b) { return U2(sub(a.x, b.x), sub(a.y, b.y)); }

Z31 OVERLOAD make_Z31(i32 a) { return (Z31) (a < 0 ? a + M31 : a); }    // Handles signed values of a
Z31 OVERLOAD make_Z31(u32 a) { return (Z31) (a); }                      // a must be in range of 0 .. M31-1
Z31 OVERLOAD make_Z31(i64 a) { return modM31q(a); }                     // Handles range -2^61..2^61

u32 get_Z31(Z31 a) { return a == M31 ? 0 : a; }                         // Get value in range 0 to M31-1
i32 get_balanced_Z31(Z31 a) { return (a & 0xC0000000) ? (i32) a - M31 : (i32) a; }  // Get balanced value in range -M31/2 to M31/2

// Assumes k reduced mod 31.
Z31 OVERLOAD shr(Z31 a, u32 k) { return (a >> k) + ((a << (31 - k)) & M31); }
GF31 OVERLOAD shr(GF31 a, u32 k) { return U2(shr(a.x, k), shr(a.y, k)); }
Z31 OVERLOAD shl(Z31 a, u32 k) { return shr(a, 31 - k); }
GF31 OVERLOAD shl(GF31 a, u32 k) { return U2(shl(a.x, k), shl(a.y, k)); }

Z31 OVERLOAD mul(Z31 a, Z31 b) { u64 t = a * (u64) b; return modM31(add((Z31)(t & M31), (Z31)(t >> 31))); }

// Multiply by 2
Z31 OVERLOAD mul2(Z31 a) { return add(a, a); }
GF31 OVERLOAD mul2(GF31 a) { return U2(mul2(a.x), mul2(a.y)); }

// Return conjugate of a
GF31 OVERLOAD conjugate(GF31 a) { return U2(a.x, neg(a.y)); }

// Complex square.  input, output 31 bits. Uses (a + i*b)^2 == ((a+b)*(a-b) + i*2*a*b).
GF31 OVERLOAD csq(GF31 a) {
  u64 r = (a.x + a.y) * (u64) (a.x + neg(a.y));      // 64-bit value, max = FFFF FFFE 0000 0004 (actually cannot exceed 9000 0000 0000 0000)
  u64 i = (a.x + a.x) * (u64) a.y;                   // 63-bit value, max = 7FFF FFFE 0000 0002
  return U2(modM31(r), modM31(i));
}

// a^2 + c
GF31 OVERLOAD csq_add(GF31 a, GF31 c) {
  u64 r = mad32(a.x + a.y, a.x + neg(a.y), c.x);      // 64-bit value, mul max = FFFF FFFE 0000 0004 (actually cannot exceed 9000 0000 0000 0000)
  u64 i = mad32(a.x + a.x, a.y, c.y);                 // 63-bit value, mul max = 7FFF FFFE 0000 0002
  return U2(modM31(r), modM31(i));
}

// a^2 - c
GF31 OVERLOAD csq_sub(GF31 a, GF31 c) {
  u64 r = mad32(a.x + a.y, a.x + neg(a.y), neg(c.x)); // 64-bit value, mul max = FFFF FFFE 0000 0004 (actually cannot exceed 9000 0000 0000 0000)
  u64 i = mad32(a.x + a.x, a.y, neg(c.y));            // 63-bit value, mul max = 7FFF FFFE 0000 0002
  return U2(modM31(r), modM31(i));
}

// a^2 + i*c
GF31 OVERLOAD csq_addi(GF31 a, GF31 c) {
  u64 r = mad32(a.x + a.y, a.x + neg(a.y), neg(c.y)); // 64-bit value, mul max = FFFF FFFE 0000 0004 (actually cannot exceed 9000 0000 0000 0000)
  u64 i = mad32(a.x + a.x, a.y, c.x);                 // 63-bit value, mul max = 7FFF FFFE 0000 0002
  return U2(modM31(r), modM31(i));
}

// a^2 - i*c
GF31 OVERLOAD csq_subi(GF31 a, GF31 c) {
  u64 r = mad32(a.x + a.y, a.x + neg(a.y), c.y);      // 64-bit value, mul max = FFFF FFFE 0000 0004 (actually cannot exceed 9000 0000 0000 0000)
  u64 i = mad32(a.x + a.x, a.y, neg(c.x));            // 63-bit value, max = 7FFF FFFE 0000 0002
  return U2(modM31(r), modM31(i));
}

// Complex mul
#if 0                                           // One less negation, requires signed shifts.  Seems microscopically faster on TitanV.
GF31 OVERLOAD cmul(GF31 a, GF31 b) {
  u64 k1 = b.x * (u64) (a.x + a.y);             // 63-bit value, max = 7FFF FFFE 0000 0002
  u64 k2 = a.x * (u64) (b.y + neg(b.x));
  u64 k3 = a.y * (u64) (b.y + b.x);
  i64 k1k3 = k1 - k3;                           // signed 63-bit value, absolute value <= 7FFF FFFE 0000 0002
  u64 k1k2 = k1 + k2;                           // unsigned 64-bit value, max = FFFF FFFC 0000 0004
  return U2(modM31(k1k3), modM31(k1k2));
}
#else
GF31 OVERLOAD cmul(GF31 a, GF31 b) {
  u32 negbx = neg(b.x);                         // Negate and add b values as much as possible in case b is used several times (as in a chainmul)
  u64 k1 = b.x * (u64)(a.x + a.y);              // 63-bit value, max = 7FFF FFFE 0000 0002
  u64 k1k2 = mad32(a.x, b.y + negbx, k1);       // unsigned 64-bit value, max = FFFF FFFC 0000 0004
  u64 k1k3 = mad32(a.y, neg(b.y) + negbx, k1);  // unsigned 64-bit value, max = FFFF FFFC 0000 0004
  return U2(modM31(k1k3), modM31(k1k2));
}
#endif

// Square a root of unity complex number
GF31 OVERLOAD csqTrig(GF31 a) { u32 two_ay = a.y + a.y; return U2(modM31(mad32(two_ay, neg(a.y), (u32)1)), modM31(a.x * (u64)two_ay)); }

// Cube w, a root of unity complex number, given w^2 and w
GF31 OVERLOAD ccubeTrig(GF31 sq, GF31 w) { u32 tmp = sq.y + sq.y; return U2(modM31(mad32(tmp, neg(w.y), w.x)), modM31(mad32(tmp, w.x, neg(w.y)))); }

// mul with (0, 1). (twiddle of tau/4, sqrt(-1) aka "i").
GF31 OVERLOAD mul_t4(GF31 a) { return U2(neg(a.y), a.x); }                                              // GWBUG:  Can caller use a version that does not negate real?

// mul with (2^15, 2^15). (twiddle of tau/8 aka sqrt(i)). Note: 2 * (+/-2^15)^2 == 1 (mod M31).
GF31 OVERLOAD mul_t8(GF31 a) { return U2(shl(sub(a.x, a.y), 15), shl(add(a.x, a.y), 15)); }

// mul with (-2^15, 2^15). (twiddle of 3*tau/8).
GF31 OVERLOAD mul_3t8(GF31 a) { return U2(shl(neg(add(a.x, a.y)), 15), shl(sub(a.x, a.y), 15)); }

// Return a+b and a-b
void OVERLOAD X2_internal(GF31 *a, GF31 *b) { GF31 t = *a; *a = add(t, *b); *b = sub(t, *b); }

// Same as X2(a, conjugate(b))
void OVERLOAD X2conjb_internal(GF31 *a, GF31 *b) { GF31 t = *a; a->x = add(a->x, b->x); a->y = sub(a->y, b->y); b->x = sub(t.x, b->x); b->y = add(t.y, b->y); }

// Same as X2(a, b), b = mul_t4(b)
void OVERLOAD X2_mul_t4_internal(GF31 *a, GF31 *b) { GF31 t = *a; *a = add(*a, *b); t.x = sub(t.x, b->x); b->x = sub(b->y, t.y); b->y = t.x; }

// Same as X2(a, b), b = mul_t8(b)
void OVERLOAD X2_mul_t8_internal(GF31 *a, GF31 *b) { X2(*a, *b); *b = mul_t8(*b); }

// Same as X2(a, b), b = mul_3t8(b)
void OVERLOAD X2_mul_3t8_internal(GF31 *a, GF31 *b) { GF31 t = *a; *a = add(t, *b); t = sub(*b, t); *b = shl(U2(add(t.x, t.y), sub(t.y, t.x)), 15); }

// Same as X2(a, b), b = conjugate(b)
void OVERLOAD X2_conjb_internal(GF31 *a, GF31 *b) { GF31 t = *a; *a = add(t, *b); b->x = sub(t.x, b->x); b->y = sub(b->y, t.y); }

void OVERLOAD SWAP_internal(GF31 *a, GF31 *b) { GF31 t = *a; *a = *b; *b = t; }

GF31 OVERLOAD addsub(GF31 a) { return U2(add(a.x, a.y), sub(a.x, a.y)); }
GF31 OVERLOAD foo2(GF31 a, GF31 b) { a = addsub(a); b = addsub(b); return addsub(U2(mul(RE(a), RE(b)), mul(IM(a), IM(b)))); }
GF31 OVERLOAD foo(GF31 a) { return foo2(a, a); }

#endif

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

// bits in reduced mod M.
#define M61 ((((Z61) 1) << 61) - 1)

Z61 OVERLOAD make_Z61(i32 a) { return (Z61) (a < 0 ? (i64) a + M61 : (i64) a); }  // Handles all values of a
Z61 OVERLOAD make_Z61(i64 a) { return (Z61) (a < 0 ? a + M61 : a); }              // a must be in range of -M61 .. M61-1
Z61 OVERLOAD make_Z61(u32 a) { return (Z61) (a); }                                // Handles all values of a
Z61 OVERLOAD make_Z61(u64 a) { return (Z61) (a); }                                // a must be in range of 0 .. M61-1

#if 0   // Slower version that keeps results strictly in the range 0 .. M61-1

u64 OVERLOAD get_Z61(Z61 a) { return a; }  // Get value in range 0 to M61-1
i64 OVERLOAD get_balanced_Z61(Z61 a) { return (hi32(a) & 0xF0000000) ? (i64) a - (i64) M61 : (i64) a; }  // Get balanced value in range -M61/2 to M61/2

Z61 OVERLOAD neg(Z61 a) { return a == 0 ? 0 : M61 - a; }                // GWBUG: Examine all callers to see if neg call can be avoided
GF61 OVERLOAD neg(GF61 a) { return U2(neg(a.x), neg(a.y)); }

Z61 OVERLOAD add(Z61 a, Z61 b) { Z61 t = a + b; Z61 m = t - M61; return (m & 0x8000000000000000ULL) ? t : m; }
//Z61 OVERLOAD add(Z61 a, Z61 b) { Z61 t = a + b; Z61 m = t - M61; return t < m ? t : m; }      // Slower on TitanV
//Z61 OVERLOAD add(Z61 a, Z61 b) { Z61 t = a + b; return t - (t >= M61 ? M61 : 0); }            // Slower on TitanV
GF61 OVERLOAD add(GF61 a, GF61 b) { return U2(add(a.x, b.x), add(a.y, b.y)); }

Z61 OVERLOAD sub(Z61 a, Z61 b) { Z61 t = a - b; return t + (((i64) t >> 63) & 0x1FFFFFFFFFFFFFFFULL); }
//Z61 OVERLOAD sub(Z61 a, Z61 b) { Z61 t = a - b; Z61 p = t + M61; return (t & 0x8000000000000000ULL) ? p : t; }  // Better???
//Z61 OVERLOAD sub(Z61 a, Z61 b) { Z61 t = a - b; return t + (t >= M61 ? M61 : 0); }            // Slower on TitanV
//  BETTER???:   t = a - b;  carry_mask = sbb x, x; (generates 32 bits of 0 or 1; return t + make_carry_mask_64_bits
GF61 OVERLOAD sub(GF61 a, GF61 b) { return U2(sub(a.x, b.x), sub(a.y, b.y)); }

// Assumes k reduced mod 61.
Z61 OVERLOAD shl(Z61 a, u32 k) { return ((a << k) + (a >> (61 - k))) & M61; }                   //GWBUG: Make sure & M61 operates on just one u32
GF61 OVERLOAD shl(GF61 a, u32 k) { return U2(shl(a.x, k), shl(a.y, k)); }
Z61 OVERLOAD shr(Z61 a, u32 k) { return ((a >> k) + (a << (61 - k))) & M61; }                   //GWBUG: Make sure & M61 operates on just one u32.  & M61 not needed?
GF61 OVERLOAD shr(GF61 a, u32 k) { return U2(shr(a.x, k), shr(a.y, k)); }

ulong2 wideMul(u64 ab, u64 cd) {
  u128 r = mul64(ab, cd);
  return U2(u128_lo64(r), u128_hi64(r));
}

Z61 OVERLOAD mul(Z61 a, Z61 b) {
  ulong2 ab = wideMul(a, b);
  u64 lo = ab.x, hi = ab.y;
  u64 lo61 = lo & M61, hi61 = (hi << 3) + (lo >> 61);
  return add(lo61, hi61);
}

Z61 OVERLOAD fma(Z61 a, Z61 b, Z61 c) { return add(mul(a, b), c); }             // GWBUG:  Can we do better?

// Multiply by 2
Z61 OVERLOAD mul2(Z61 a) { return ((a + a) + (a >> 60)) & M61; }        // GWBUG: Make sure "+ a>>60" does an add to lower u32 without a followup adc.
GF61 OVERLOAD mul2(GF61 a) { return U2(mul2(a.x), mul2(a.y)); }

// Return conjugate of a
GF61 OVERLOAD conjugate(GF61 a) { return U2(a.x, neg(a.y)); }

// Complex square.  input, output 61 bits. Uses (a + i*b)^2 == ((a+b)*(a-b) + i*2*a*b).
GF61 OVERLOAD csq(GF61 a) { return U2(mul(add(a.x, a.y), sub(a.x, a.y)), mul2(mul(a.x, a.y))); }        //GWBUG: Probably faster to double a.y and have a mul that takes non-normalized inputs

// a^2 + c
GF61 OVERLOAD csqa(GF61 a, GF61 c) { return add(csq(a), c); }                                           // GWBUG: inline csq so we only "mod" after adding c??  Find a way to use fma instructions

// Complex mul
//GF61 OVERLOAD cmul(GF61 a, GF61 b) { return U2(sub(mul(a.x, b.x), mul(a.y, b.y)), add(mul(a.x, b.y), mul(a.y, b.x)));}   // GWBUG:  Is a 3 multiply complex mul faster?  See above
GF61 OVERLOAD cmul(GF61 a, GF61 b) {
  Z61 k1 = mul(b.x, add(a.x, a.y));
  Z61 k2 = mul(a.x, sub(b.y, b.x));
  Z61 k3 = mul(a.y, add(b.y, b.x));
  return U2(sub(k1, k3), add(k1, k2));
}

// mul with (0, 1). (twiddle of tau/4, sqrt(-1) aka "i").
GF61 OVERLOAD mul_t4(GF61 a) { return U2(neg(a.y), a.x); }                                              // GWBUG:  Can caller use a version that does not negate real?

// mul with (-2^30, -2^30). (twiddle of tau/8 aka sqrt(i)). Note: 2 * (+/-2^30)^2 == 1 (mod M61).
GF61 OVERLOAD mul_t8(GF61 a) { return shl(U2(sub(a.y, a.x), neg(add(a.x, a.y))), 30); }       // GWBUG:  Can caller use a version that does not negate real?

// mul with (2^30, -2^30). (twiddle of 3*tau/8).
GF61 OVERLOAD mul_3t8(GF61 a) { return shl(U2(add(a.x, a.y), sub(a.y, a.x)), 30); }

// Return a+b and a-b
void OVERLOAD X2_internal(GF61 *a, GF61 *b) { GF61 t = *a; *a = add(t, *b); *b = sub(t, *b); }

// Same as X2(a, conjugate(b))
void OVERLOAD X2conjb_internal(GF61 *a, GF61 *b) { GF61 t = *a; a->x = add(a->x, b->x); a->y = sub(a->y, b->y); b->x = sub(t.x, b->x); b->y = add(t.y, b->y); }

// Same as X2(a, b), b = mul_t4(b)
void OVERLOAD X2_mul_t4_internal(GF61 *a, GF61 *b) { GF61 t = *a; *a = add(*a, *b); t.x = sub(t.x, b->x); b->x = sub(b->y, t.y); b->y = t.x; }

// Same as X2(a, b), b = mul_t8(b)
void OVERLOAD X2_mul_t8_internal(GF61 *a, GF61 *b) { X2(*a, *b); *b = mul_t8(*b); }

// Same as X2(a, b), b = mul_3t8(b)
void OVERLOAD X2_mul_3t8_internal(GF61 *a, GF61 *b) { X2(*a, *b); *b = mul_3t8(*b); }

// Same as X2(a, b), b = conjugate(b)
void OVERLOAD X2_conjb_internal(GF61 *a, GF61 *b) { GF61 t = *a; *a = add(t, *b); b->x = sub(t.x, b->x); b->y = sub(b->y, t.y); }

void OVERLOAD SWAP_internal(GF61 *a, GF61 *b) { GF61 t = *a; *a = *b; *b = t; }

GF61 OVERLOAD addsub(GF61 a) { return U2(add(a.x, a.y), sub(a.x, a.y)); }
GF61 OVERLOAD foo2(GF61 a, GF61 b) { a = addsub(a); b = addsub(b); return addsub(U2(mul(RE(a), RE(b)), mul(IM(a), IM(b)))); }
GF61 OVERLOAD foo(GF61 a) { return foo2(a, a); }

// The following routines can be used to reduce mod M61 operations (in the other Z61 implementations).
// Caller must track how many M61s need to be added to make positive values for subtractions.
// In function names, "q" stands for quick, "s" stands for slow (i.e. does mod).
// These functions are untested with this strict Z61 implementation.  Callers need to eliminate all uses of + or - operators.

Z61 OVERLOAD modM61(Z61 a) { return a; }
GF61 OVERLOAD modM61(GF61 a) { return a; }
Z61 OVERLOAD neg(Z61 a, u32 m61_count) { return neg(a); }
GF61 OVERLOAD neg(GF61 a, u32 m61_count) { return neg(a); }
Z61 OVERLOAD addq(Z61 a, Z61 b) { return add(a, b); }
GF61 OVERLOAD addq(GF61 a, GF61 b) { return add(a, b); }
Z61 OVERLOAD subq(Z61 a, Z61 b, u32 m61_count) { return sub(a, b); }
GF61 OVERLOAD subq(GF61 a, GF61 b, u32 m61_count) { return sub(a, b); }
Z61 OVERLOAD subs(Z61 a, Z61 b, u32 m61_count) { return sub(a, b); }
GF61 OVERLOAD subs(GF61 a, GF61 b, u32 m61_count) { return sub(a, b); }
void OVERLOAD X2q(GF61 *a, GF61 *b, u32 m61_count) { X2_internal(a, b); }
void OVERLOAD X2q_mul_t4(GF61 *a, GF61 *b, u32 m61_count) { X2_mul_t4_internal(a, b); }
void OVERLOAD X2s(GF61 *a, GF61 *b, u32 m61_count) { X2_internal(a, b); }
void OVERLOAD X2s_conjb(GF61 *a, GF61 *b, u32 m61_count) { X2_conjb_internal(a, b); }




// Philosophy: This Z61/GF61 implementation uses faster, sloppier mod M61 reduction where the end result is in the range 0..M61+epsilon.
// This implementation also handles subtractions by adding enough M61s to make a value positive.  This allows us to always deal with positive
// intermediate results.  The downside is that a caller using the sloppy/quick routines must keep track of how large unreduced values can get.
// An alternative implementation is to have Z61 be an i64 (costs us a precious bit of precision) but is surprisingly slower (at least on TitanV) because
//      mod(a - b),             where the mod routinue uses a signed right shift is slower than
//      mod(a + (M61*2 - b))    where the mod routine uses an unsigned shift right.
// However, a long string of subtracts (example, fft8 does 3 subtracts before mod M61 might be better off using negative intermediate results.
// The mul routine (and obviously csq and cmul) must use only positive values as __int128 multiply is very slow.

#elif 1   // Faster version that keeps results in the range 0 .. M61+epsilon

u64 OVERLOAD get_Z61(Z61 a) { Z61 m = a - M61; return (m & 0x8000000000000000ULL) ? a : m; }  // Get value in range 0 to M61-1
i64 OVERLOAD get_balanced_Z61(Z61 a) { return (a >= 0x1000000000000000ULL) ? (i64) a - (i64) M61 : (i64) a; }  // Get balanced value in range -M61/2 to M61/2

// Internal routine to bring Z61 value into the range 0..M61+epsilon
Z61 OVERLOAD modM61(Z61 a) { return (a & M61) + (a >> 61); }
GF61 OVERLOAD modM61(GF61 a) { return U2(modM61(a.x), modM61(a.y)); }
// Internal routine to negate a value by adding the specified number of M61s -- no mod M61 reduction
Z61 OVERLOAD neg(Z61 a, const u32 m61_count) { return m61_count * M61 - a; }
GF61 OVERLOAD neg(GF61 a, const u32 m61_count) { return U2(neg(a.x, m61_count), neg(a.y, m61_count)); }

Z61 OVERLOAD add(Z61 a, Z61 b) { return modM61(a + b); }
GF61 OVERLOAD add(GF61 a, GF61 b) { return U2(add(a.x, b.x), add(a.y, b.y)); }

Z61 OVERLOAD sub(Z61 a, Z61 b) { return modM61(a + neg(b, 2)); }
GF61 OVERLOAD sub(GF61 a, GF61 b) { return U2(sub(a.x, b.x), sub(a.y, b.y)); }

Z61 OVERLOAD neg(Z61 a) { return modM61(neg(a, 2)); }                // GWBUG: Examine all callers to see if neg call can be avoided
GF61 OVERLOAD neg(GF61 a) { return U2(neg(a.x), neg(a.y)); }

// Assumes k reduced mod 61.
Z61 OVERLOAD shr(Z61 a, u32 k) { return (a >> k) + ((a << (61 - k)) & M61); }         // Return range 0..M61+2^(61-k), can handle 64-bit inputs but small k is big epsilon
GF61 OVERLOAD shr(GF61 a, u32 k) { return U2(shr(a.x, k), shr(a.y, k)); }
Z61 OVERLOAD shl(Z61 a, u32 k) { return shr(a, 61 - k); }         // Return range 0..M61+2^k, can handle 64-bit inputs but large k yields big epsilon
//Z61 OVERLOAD shl(Z61 a, u32 k) { return modM61(a << k) + ((a >> (64 - k)) << 3); }         // Return range 0..M61+2^k, can handle 64-bit inputs but large k is big epsilon
//Z61 OVERLOAD shl(Z61 a, u32 k) { return modM61((a << k) + ((a >> (64 - k)) << 3)); }       // Return range 0..M61+epsilon, input must be M61+epsilon a full 62-bit value can overflow
GF61 OVERLOAD shl(GF61 a, u32 k) { return U2(shl(a.x, k), shl(a.y, k)); }

ulong2 wideMul(u64 ab, u64 cd) {
  u128 r = mul64(ab, cd);
  return U2(u128_lo64(r), u128_hi64(r));
}

// Returns a * b not modded by M61.  Max value of result depends on the m61_counts of the inputs.
// Let n = (a_m61_count - 1) * (b_m61_count - 1).  This is the maximum value in the highest 6 bits of a * b.
// If n <= 4 result will be at most (n+1)*M61+epsilon.
// If n > 4 result will be at most 2*M61+epsilon.
Z61 OVERLOAD weakMul(Z61 a, Z61 b, const u32 a_m61_count, const u32 b_m61_count) {
  ulong2 ab = wideMul(a, b);
  u64 lo = ab.x, hi = ab.y;
  u64 lo61 = lo & M61;                                  // Max value is M61
  if ((a_m61_count - 1) * (b_m61_count - 1) <= 4) {
     hi = (hi << 3) + (lo >> 61);                       // Max value is (a_m61_count - 1) * (b_m61_count - 1) * M61 + epsilon
     return lo61 + hi;                                  // Max value is ((a_m61_count - 1) * (b_m61_count - 1) + 1) * M61 + epsilon
  } else {
     u64 hi61 = ((hi << 3) + (lo >> 61)) & M61;         // Max value is M61
     return lo61 + hi61 + (hi >> 58);                   // Max value is 2*M61 + epsilon
  }
}

Z61 OVERLOAD mul(Z61 a, Z61 b) { return modM61(weakMul(a, b, 2, 2)); }

Z61 OVERLOAD fma(Z61 a, Z61 b, Z61 c) { return modM61(weakMul(a, b, 2, 2) + c); }             // GWBUG:  Can we do better?

// Multiply by 2
Z61 OVERLOAD mul2(Z61 a) { return add(a, a); }
GF61 OVERLOAD mul2(GF61 a) { return U2(mul2(a.x), mul2(a.y)); }

// Return conjugate of a
GF61 OVERLOAD conjugate(GF61 a) { return U2(a.x, neg(a.y)); }

// Complex square. Uses (a + i*b)^2 == ((a+b)*(a-b) + i*2*a*b).
GF61 OVERLOAD csqq(GF61 a, const u32 m61_count) {
  if (m61_count > 4) return csqq(modM61(a), 2);
  Z61 re = weakMul(a.x + a.y, a.x + neg(a.y, m61_count), 2 * m61_count - 1, 2 * m61_count);
  Z61 im = weakMul(a.x + a.x, a.y, 2 * m61_count - 1, m61_count);
  return U2(re, im);
}
GF61 OVERLOAD csqs(GF61 a, const u32 m61_count) { return modM61(csqq(a, m61_count)); }
GF61 OVERLOAD csq(GF61 a) { return csqs(a, 2); }

// a^2 + c
GF61 OVERLOAD csqa(GF61 a, GF61 c) { return U2(modM61(weakMul(a.x + a.y, a.x + neg(a.y, 2), 3, 4) + c.x), modM61(weakMul(a.x + a.x, a.y, 3, 2) + c.y)); }

// Complex mul
#if 0
GF61 OVERLOAD cmul(GF61 a, GF61 b) {              // Use 3-epsilon extra bits in u64
  Z61 k1 = weakMul(b.x, a.x + a.y, 2, 3);         // max value is 3*M61+epsilon
  Z61 k2 = weakMul(a.x, b.y + neg(b.x, 2), 2, 3); // max value is 3*M61+epsilon
  Z61 k3 = weakMul(a.y, b.y + b.x, 2, 3);         // max value is 3*M61+epsilon
  return U2(modM61(k1 + neg(k3, 4)), modM61(k1 + k2));
}
#else
Z61 OVERLOAD weakMulAdd(Z61 a, Z61 b, u128 c, const u32 a_m61_count, const u32 b_m61_count) {  // Max c value assumed to be 2*M61^2+epsilon
  u128 ab = mad64(a, b, c);                             // Max value is ((a_m61_count - 1) * (b_m61_count - 1) + 2) * M61^2 + epsilon
  u64 lo = u128_lo64(ab), hi = u128_hi64(ab);
  u64 lo61 = lo & M61;                                  // Max value is M61
  if ((a_m61_count - 1) * (b_m61_count - 1) + 2 <= 6) {
     hi = (hi << 3) + (lo >> 61);                       // Max value is ((a_m61_count - 1) * (b_m61_count - 1) + 2) * M61 + epsilon
     return lo61 + hi;                                  // Max value is ((a_m61_count - 1) * (b_m61_count - 1) + 3) * M61 + epsilon
  } else {
     u64 hi61 = ((hi << 3) + (lo >> 61)) & M61;         // Max value is M61
     return lo61 + hi61 + (hi >> 58);                   // Max value is 2*M61 + epsilon
  }
}
GF61 OVERLOAD cmul(GF61 a, GF61 b) {
  u128 k1 = mul64(b.x, a.x + a.y);                            // max value is 2*M61^2+epsilon
  Z61 k1k2 = weakMulAdd(a.x, b.y + neg(b.x, 2), k1, 2, 4);    // max value is 6*M61+epsilon
  Z61 k1k3 = weakMulAdd(a.y, neg(b.y + b.x, 3), k1, 2, 4);    // max value is 6*M61+epsilon
  return U2(modM61(k1k3), modM61(k1k2));
}
#endif

// Square a root of unity complex number (the second version may be faster if the compiler optimizes the u128 squaring).
//GF61 OVERLOAD csqTrig(GF61 a) { Z61 two_ay = a.y + a.y; return U2(modM61(1 + weakMul(two_ay, neg(a.y, 2))), mul(a.x, two_ay)); }
GF61 OVERLOAD csqTrig(GF61 a) { Z61 ay_sq = weakMul(a.y, a.y, 2, 2); return U2(modM61(1 + neg(ay_sq + ay_sq, 4)), mul2(weakMul(a.x, a.y, 2, 2))); }

// Cube w, a root of unity complex number, given w^2 and w
GF61 OVERLOAD ccubeTrig(GF61 sq, GF61 w) { Z61 tmp = sq.y + sq.y; return U2(modM61(weakMul(tmp, neg(w.y, 2), 3, 3) + w.x), modM61(weakMul(tmp, w.x, 3, 2) + neg(w.y, 2))); }

// a + i*b 
GF61 OVERLOAD addi(GF61 a, GF61 b) { return U2(sub(a.x, b.y), add(a.y, b.x)); }
// a - i*b 
GF61 OVERLOAD subi(GF61 a, GF61 b) { return U2(add(a.x, b.y), sub(a.y, b.x)); }

// mul with (0, 1). (twiddle of tau/4, sqrt(-1) aka "i").
GF61 OVERLOAD mul_t4(GF61 a) { return U2(neg(a.y), a.x); }                                      // GWBUG:  Can caller use a version that does not negate real?

// mul with (-2^30, -2^30). (twiddle of tau/8 aka sqrt(i)). Note: 2 * (+/-2^30)^2 == 1 (mod M61).
GF61 OVERLOAD mul_t8(GF61 a, const u32 m61_count) { return shl(U2(a.y + neg(a.x, m61_count), neg(a.x + a.y, 2 * m61_count - 1)), 30); }
GF61 OVERLOAD mul_t8(GF61 a) { return mul_t8(a, 2); }

// mul with (2^30, -2^30). (twiddle of 3*tau/8).
GF61 OVERLOAD mul_3t8(GF61 a, const u32 m61_count) { return shl(U2(a.x + a.y, a.y + neg(a.x, m61_count)), 30); }
GF61 OVERLOAD mul_3t8(GF61 a) { return mul_3t8(a, 2); }

// Return a+b and a-b
void OVERLOAD X2_internal(GF61 *a, GF61 *b) { GF61 t = *a; *a = add(t, *b); *b = sub(t, *b); }

// Same as X2(a, conjugate(b))
void OVERLOAD X2conjb_internal(GF61 *a, GF61 *b) { GF61 t = *a; a->x = add(a->x, b->x); a->y = sub(a->y, b->y); b->x = sub(t.x, b->x); b->y = add(t.y, b->y); }

// Same as X2(a, b), b = mul_t4(b)
void OVERLOAD X2_mul_t4_internal(GF61 *a, GF61 *b) { GF61 t = *a; *a = add(*a, *b); t.x = sub(t.x, b->x); b->x = sub(b->y, t.y); b->y = t.x; }

// Same as X2(a, b), b = mul_t8(b)
void OVERLOAD X2_mul_t8_internal(GF61 *a, GF61 *b) { GF61 t = *a; *a = add(t, *b); t = *b + neg(t, 2); *b = shl(U2(t.x + neg(t.y, 4), t.x + t.y), 30); }

// Same as X2(a, b), b = mul_3t8(b)
void OVERLOAD X2_mul_3t8_internal(GF61 *a, GF61 *b) { GF61 t = *a; *a = add(t, *b); *b = t + neg(*b, 2); *b = mul_3t8(*b, 4); }

// Same as X2(a, b), b = conjugate(b)
void OVERLOAD X2_conjb_internal(GF61 *a, GF61 *b) { GF61 t = *a; *a = add(t, *b); b->x = sub(t.x, b->x); b->y = sub(b->y, t.y); }

void OVERLOAD SWAP_internal(GF61 *a, GF61 *b) { GF61 t = *a; *a = *b; *b = t; }

GF61 OVERLOAD addsub(GF61 a) { return U2(add(a.x, a.y), sub(a.x, a.y)); }
GF61 OVERLOAD foo2(GF61 a, GF61 b) { a = addsub(a); b = addsub(b); return addsub(U2(mul(RE(a), RE(b)), mul(IM(a), IM(b)))); }
GF61 OVERLOAD foo(GF61 a) { return foo2(a, a); }

// The following routines can be used to reduce mod M61 operations.  Caller must track how many M61s need to be added to make positive
// values for subtractions.  In function names, "q" stands for quick (no modM61), "s" stands for slow (i.e. does modM61).

Z61 OVERLOAD addq(Z61 a, Z61 b) { return a + b; }
GF61 OVERLOAD addq(GF61 a, GF61 b) { return U2(addq(a.x, b.x), addq(a.y, b.y)); }

Z61 OVERLOAD subq(Z61 a, Z61 b, const u32 m61_count) { return a + neg(b, m61_count); }
GF61 OVERLOAD subq(GF61 a, GF61 b, const u32 m61_count) { return U2(subq(a.x, b.x, m61_count), subq(a.y, b.y, m61_count)); }

Z61 OVERLOAD subs(Z61 a, Z61 b, const u32 m61_count) { return modM61(a + neg(b, m61_count)); }
GF61 OVERLOAD subs(GF61 a, GF61 b, const u32 m61_count) { return U2(subs(a.x, b.x, m61_count), subs(a.y, b.y, m61_count)); }

GF61 OVERLOAD addiq(GF61 a, GF61 b, const u32 m61_count) { return U2(subq(a.x, b.y, m61_count), addq(a.y, b.x)); }
GF61 OVERLOAD subiq(GF61 a, GF61 b, const u32 m61_count) { return U2(addq(a.x, b.y), subq(a.y, b.x, m61_count)); }

void OVERLOAD X2q(GF61 *a, GF61 *b, const u32 m61_count) { GF61 t = *a; *a = t + *b; *b = t + neg(*b, m61_count); }
void OVERLOAD X2q_mul_t4(GF61 *a, GF61 *b, const u32 m61_count) { GF61 t = *a; *a = t + *b; t.x = t.x + neg(b->x, m61_count); b->x = b->y + neg(t.y, m61_count); b->y = t.x; }
void OVERLOAD X2q_mul_t8(GF61 *a, GF61 *b, const u32 m61_count) { GF61 t = *a; *a = t + *b; t = *b + neg(t, m61_count); *b = shl(U2(t.x + neg(t.y, m61_count * 2), t.x + t.y), 30); }
void OVERLOAD X2q_mul_3t8(GF61 *a, GF61 *b, const u32 m61_count) { GF61 t = *a; *a = t + *b; t = t + neg(*b, m61_count); *b = mul_3t8(t, m61_count * 2); }

void OVERLOAD X2s(GF61 *a, GF61 *b, const u32 m61_count) { GF61 t = *a; *a = add(t, *b); *b = subs(t, *b, m61_count); }
void OVERLOAD X2s_conjb(GF61 *a, GF61 *b, const u32 a_m61_count, const u32 b_m61_count) { GF61 t = *a; *a = add(t, *b); b->x = subs(t.x, b->x, b_m61_count); b->y = subs(b->y, t.y, a_m61_count); }

#endif

#endif

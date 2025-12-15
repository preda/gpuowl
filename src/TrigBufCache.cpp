// Copyright Mihai Preda

#include <cstring>
#include "TrigBufCache.h"

#define SAVE_ONE_MORE_WIDTH_MUL  0      // I want to make saving the only option -- but rocm optimizer is inexplicably making it slower in carryfused
#define SAVE_ONE_MORE_HEIGHT_MUL 1      // In tailSquare this is the fastest option

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#ifndef M_PI
#define M_PI 3.1415926535897931
#endif

static_assert(sizeof(double2) == 16, "size double2");

// For small angles, return "fancy" cos - 1 for increased precision
double2 root1Fancy(u32 N, u32 k) {
  assert(!(N&7));
  assert(k < N);
  assert(k < N/4);

  long double angle = M_PIl * k / (N / 2);
  return {double(cosl(angle) - 1), double(sinl(angle))};
}

static double trigNorm(double c, double s) { return c * c + s * s; }
static double trigError(double c, double s) { return abs(trigNorm(c, s) - 1.0); }

// Round trig long double to double as to satisfy c^2 + s^2 == 1 as best as possible
static double2 roundTrig(long double lc, long double ls) {
  double c1 = lc;
  double c2 = nexttoward(c1, lc);
  double s1 = ls;
  double s2 = nexttoward(s1, ls);

  double c = c1;
  double s = s1;
  for (double tryC : {c1, c2}) {
    for (double tryS : {s1, s2}) {
      if (trigError(tryC, tryS) < trigError(c, s)) {
        c = tryC;
        s = tryS;
      }
    }
  }
  return {c, s};
}

// Returns the primitive root of unity of order N, to the power k.
double2 root1(u32 N, u32 k) {
  assert(k < N);
  if (k >= N/2) {
    auto [c, s] = root1(N, k - N/2);
    return {-c, -s};
  } else if (k > N/4) {
    auto [c, s] = root1(N, N/2 - k);
    return {-c, s};
  } else if (k > N/8) {
    auto [c, s] = root1(N, N/4 - k);
    return {s, c};
  } else {
    assert(k <= N/8);

    long double angle = M_PIl * k / (N / 2);

#if 1
    return roundTrig(cosl(angle), sinl(angle));
#else
    double c = cos(double(angle)), s = sin(double(angle));
    if ((c * c + s * s == 1.0)) {
      return {c, s};
    } else {
      return {double(cosl(angle)), double(sinl(angle))};
    }
#endif
  }
}

// Epsilon value, 2^-250, should have an exact representation as a double.  Used to avoid divide-by-zero in root1over.
const double epsilon = 5.5271478752604445602472651921923E-76;  // Protect against divide by zero

// Returns the primitive root of unity of order N, to the power k.  Returned format is cosine, sine/cosine.
double2 root1over(u32 N, u32 k) {
  assert(k < N);

  long double angle = M_PIl * k / (N / 2);
  double c = cos(angle);
  long double s = sinl(angle);

  if (c > -1.0e-15 && c < 1.0e-15) c = epsilon;
  s = s / c;
  return {c, double(s)};
}

// Returns the primitive root of unity of order N, to the power k.  Returns only the cosine value.
double root1cos(u32 N, u32 k) {
  assert(k < N);

  long double angle = M_PIl * k / (N / 2);
  double c = cos(angle);

  if (c > -1.0e-15 && c < 1.0e-15) c = epsilon;
  return c;
}

// Returns the primitive root of unity of order N, to the power k.  Returns only the cosine value divided by another cosine value.
double root1cosover(u32 N, u32 k, double over) {
  assert(k < N);

  long double angle = M_PIl * k / (N / 2);
  long double c = cosl(angle);

  if (c > -1.0e-15 && c < 1.0e-15) c = epsilon;
  return double(c / over);
}

static const constexpr bool LOG_TRIG_ALLOC = false;

// Interleave two lines of trig values so that AMD GPUs can use global_load_dwordx4 instructions
void T2shuffle(u32 size, u32 radix, u32 line, vector<double> &tab) {
  vector<double> line1, line2;
  u32 line_size = size / radix;
  for (u32 col = 0; col < line_size; ++col) {
    line1.push_back(tab[line*line_size + col]);
    line2.push_back(tab[(line+1)*line_size + col]);
  }
  for (u32 col = 0; col < line_size; ++col) {
    tab[line*line_size + 2*col] = line1[col];
    tab[line*line_size + 2*col + 1] = line2[col];
  }
}

vector<double2> genSmallTrigFP64(u32 size, u32 radix) {
  if (LOG_TRIG_ALLOC) { log("genSmallTrigFP64(%u, %u)\n", size, radix); }
  u32 WG = size / radix;
  vector<double2> tab;

// old fft_WIDTH and fft_HEIGHT
  for (u32 line = 1; line < radix; ++line) {
    for (u32 col = 0; col < WG; ++col) {
      tab.push_back(radix / line >= 8 ? root1Fancy(size, col * line) : root1(size, col * line));
    }
  }
  tab.resize(size);

// New fft_WIDTH and fft_HEIGHT
// We need two versions of trig values.  One where we save one more mul and one where we don't.
// In theory, we should always use save one more mul but the rocm optimizer is doing something weird in fft_WIDTH.

  for (u32 save_one_more_mul = 0; save_one_more_mul <= 1; ++save_one_more_mul) {
    vector<double> tab1;
    if (save_one_more_mul) tab.resize(3*size);

    // Sine/cosine values for first fft4 or fft8
    for (u32 line = 1; line < radix; ++line) {
      for (u32 col = 0; col < WG; ++col) {
        double2 root = root1over(size, col * line);
        tab1.push_back(root.second);
      }
    }

    // Sine/cosine values for later fft4 or fft8
    for (u32 line = 0; line < radix; ++line) {
      for (u32 col = 0; col < WG; col += radix) {
        double2 root = root1over(size, col * line);
        tab1.push_back(root.second);
      }
    }

    // Cosine values for first fft4 or fft8 (output in post-shufl order)
//TODO: Examine why when sine is 0.0 cosine is not 1.0 or -1.0 (printf is outputting 0.999... and -0.999...)
    for (u32 grp = 0; grp < WG; ++grp) {
      u32 line = grp / (WG/radix);  // Output "line" number, where each line multiplies a different u[i].  There are radix lines.  Each line has WG values.
      for (u32 col = 0; col < radix; ++col) {
        double divide_by = 1.0;
        // Compute cosine3 / cosine1
        if ((radix == 4 && line == 3) || (radix == 8 && save_one_more_mul && line == 3)) { 
          divide_by = root1cos(size, col * (grp - 2*(WG/radix)));
        }
        // Compute cosine5 / cosine1, cosine6 / cosine2, cosine7 / cosine3
        if (radix == 8 && ((save_one_more_mul && line == 5) || line == 6 || line == 7)) { 
          divide_by = root1cos(size, col * (grp - 4*(WG/radix)));
        }
        tab1.push_back(root1cosover(size, col * grp, divide_by));
      }
    }

    // Cosine values for later fft4 or fft8 (output in post-shufl order).  Similar to cosines above but output every radix-th value.
    for (u32 grp = 0; grp < radix; ++grp) {
      for (u32 col = 0; col < WG; col += radix) {
        u32 line = col / (WG/radix);
        double divide_by = 1.0;
        // Compute cosine3 / cosine1
        if ((radix == 4 && line == 3) || (radix == 8 && save_one_more_mul && line == 3)) { 
          divide_by = root1cos(size, grp * (col - 2*(WG/radix)));
        }
        // Compute cosine5 / cosine1, cosine6 / cosine2, cosine7 / cosine3
        if (radix == 8 && ((save_one_more_mul && line == 5) || line == 6 || line == 7)) { 
          divide_by = root1cos(size, grp * (col - 4*(WG/radix)));
        }
        tab1.push_back(root1cosover(size, grp * col, divide_by));
      }
    }

    // Interleave first fft4 or fft8 trig values for faster AMD GPU access
    for (u32 i = 0; i < radix-2; i += 2) T2shuffle(size, radix, i, tab1);
    for (u32 i = radix; i < 2*radix; i += 2) T2shuffle(size, radix, i, tab1);

    // Convert to a vector of double2
    for (u32 i = 0; i < tab1.size(); i += 2) tab.push_back({tab1[i], tab1[i+1]});
  }

  tab.resize(5*size);
  return tab;
}

// Generate the small trig values for fft_HEIGHT plus optionally trig values used in pairSq.
vector<double2> genSmallTrigComboFP64(Args *args, u32 width, u32 middle, u32 size, u32 radix, bool tail_single_wide) {
  if (LOG_TRIG_ALLOC) { log("genSmallTrigComboFP64(%u, %u)\n", size, radix); }

  vector<double2> tab = genSmallTrigFP64(size, radix);

  u32 tail_trigs = args->value("TAIL_TRIGS", 2);                   // Default is calculating from scratch, no memory accesses

  // From tailSquare pre-calculate some or all of these:  T2 trig = slowTrig_N(line + H * lowMe, ND / NH * 2);
  if (tail_trigs == 1) {          // Some trig values in memory, some are computed with a complex multiply.  Best option on a Radeon VII.
    u32 height = size;
    // Output line 0 trig values to be read by every u,v pair of lines
    for (u32 me = 0; me < height / radix; ++me) {
      tab.push_back(root1(width * middle * height, width * middle * me));
    }
    // Output the one or two T2 multipliers to be read by one u,v pair of lines
    for (u32 line = 0; line <= width * middle / 2; ++line) {
      tab.push_back(root1Fancy(width * middle * height, line));
      if (!tail_single_wide) tab.push_back(root1Fancy(width * middle * height, line ? width * middle - line : width * middle / 2));
    }
  }
  if (tail_trigs == 0) {          // All trig values read from memory.  Best option for GPUs with lousy DP performance.
    u32 height = size;
    for (u32 u = 0; u <= width * middle / 2; ++u) {
      for (u32 v = 0; v < (tail_single_wide ? 1 : 2); ++v) {
        u32 line = (v == 0) ? u : (u ? width * middle - u : width * middle / 2);
        for (u32 me = 0; me < height / radix; ++me) {
          tab.push_back(root1(width * middle * height, line + width * middle * me));
        }
      }
    }
  }

  return tab;
}

// starting from a MIDDLE of 5 we consider angles in [0, 2Pi/MIDDLE] as worth storing with the
// cos-1 "fancy" trick.
#define SHARP_MIDDLE 5

vector<double2> genMiddleTrigFP64(u32 smallH, u32 middle, u32 width) {
  if (LOG_TRIG_ALLOC) { log("genMiddleTrigFP64(%u, %u, %u)\n", smallH, middle, width); }
  vector<double2> tab;
  if (middle == 1) {
    tab.resize(1);
  } else {
    if (middle < SHARP_MIDDLE) {
      for (u32 k = 0; k < smallH; ++k) { tab.push_back(root1(smallH * middle, k)); }
      for (u32 k = 0; k < width; ++k)  { tab.push_back(root1(middle * width, k)); }
      for (u32 k = 0; k < smallH; ++k)  { tab.push_back(root1(width * middle * smallH, k)); }
    } else {
      for (u32 k = 0; k < smallH; ++k) { tab.push_back(root1Fancy(smallH * middle, k)); }
      for (u32 k = 0; k < width; ++k)  { tab.push_back(root1Fancy(middle * width, k)); }
      for (u32 k = 0; k < smallH; ++k)  { tab.push_back(root1(width * middle * smallH, k)); }
    }
  }
  return tab;
}


/**************************************************************************/
/*           Similar to above, but for an FFT based on floats             */
/**************************************************************************/

// For small angles, return "fancy" cos - 1 for increased precision
float2 root1FancyFP32(u32 N, u32 k) {
  assert(!(N&7));
  assert(k < N);
  assert(k < N/4);

  double angle = M_PI * k / (N / 2);
  return {float(cos(angle) - 1), float(sin(angle))};
}

static float trigNorm(float c, float s) { return c * c + s * s; }
static float trigError(float c, float s) { return abs(trigNorm(c, s) - 1.0f); }

// Round trig double to float as to satisfy c^2 + s^2 == 1 as best as possible
static float2 roundTrig(double lc, double ls) {
  float c1 = lc;
  float c2 = nexttoward(c1, lc);
  float s1 = ls;
  float s2 = nexttoward(s1, ls);

  float c = c1;
  float s = s1;
  for (float tryC : {c1, c2}) {
    for (float tryS : {s1, s2}) {
      if (trigError(tryC, tryS) < trigError(c, s)) {
        c = tryC;
        s = tryS;
      }
    }
  }
  return {c, s};
}

// Returns the primitive root of unity of order N, to the power k.
float2 root1FP32(u32 N, u32 k) {
  assert(k < N);
  if (k >= N/2) {
    auto [c, s] = root1FP32(N, k - N/2);
    return {-c, -s};
  } else if (k > N/4) {
    auto [c, s] = root1FP32(N, N/2 - k);
    return {-c, s};
  } else if (k > N/8) {
    auto [c, s] = root1FP32(N, N/4 - k);
    return {s, c};
  } else {
    assert(k <= N/8);

    double angle = M_PI * k / (N / 2);
    return roundTrig(cos(angle), sin(angle));
  }
}

vector<float2> genSmallTrigFP32(u32 size, u32 radix) {
  u32 WG = size / radix;
  vector<float2> tab;

// old fft_WIDTH and fft_HEIGHT
  for (u32 line = 1; line < radix; ++line) {
    for (u32 col = 0; col < WG; ++col) {
      tab.push_back(radix / line >= 8 ? root1FancyFP32(size, col * line) : root1FP32(size, col * line));
    }
  }
  tab.resize(size);
  return tab;
}

// Generate the small trig values for fft_HEIGHT plus optionally trig values used in pairSq.
vector<float2> genSmallTrigComboFP32(Args *args, u32 width, u32 middle, u32 size, u32 radix, bool tail_single_wide) {
  vector<float2> tab = genSmallTrigFP32(size, radix);

  u32 tail_trigs = args->value("TAIL_TRIGS32", 2);          // Default is calculating from scratch, no memory accesses

  // From tailSquare pre-calculate some or all of these:  F2 trig = slowTrig_N(line + H * lowMe, ND / NH * 2);
  if (tail_trigs == 1) {          // Some trig values in memory, some are computed with a complex multiply.
    u32 height = size;
    // Output line 0 trig values to be read by every u,v pair of lines
    for (u32 me = 0; me < height / radix; ++me) {
      tab.push_back(root1FP32(width * middle * height, width * middle * me));
    }
    // Output the one or two F2 multipliers to be read by one u,v pair of lines
    for (u32 line = 0; line <= width * middle / 2; ++line) {
      tab.push_back(root1FancyFP32(width * middle * height, line));
      if (!tail_single_wide) tab.push_back(root1FancyFP32(width * middle * height, line ? width * middle - line : width * middle / 2));
    }
  }
  if (tail_trigs == 0) {          // All trig values read from memory.  Best option for GPUs with lousy FP performance?
    u32 height = size;
    for (u32 u = 0; u <= width * middle / 2; ++u) {
      for (u32 v = 0; v < (tail_single_wide ? 1 : 2); ++v) {
        u32 line = (v == 0) ? u : (u ? width * middle - u : width * middle / 2);
        for (u32 me = 0; me < height / radix; ++me) {
          tab.push_back(root1FP32(width * middle * height, line + width * middle * me));
        }
      }
    }
  }

  return tab;
}

vector<float2> genMiddleTrigFP32(u32 smallH, u32 middle, u32 width) {
  vector<float2> tab;
  if (middle == 1) {
    tab.resize(1);
  } else {
    if (middle < SHARP_MIDDLE) {
      for (u32 k = 0; k < smallH; ++k) { tab.push_back(root1FP32(smallH * middle, k)); }
      for (u32 k = 0; k < width; ++k)  { tab.push_back(root1FP32(middle * width, k)); }
      for (u32 k = 0; k < smallH; ++k)  { tab.push_back(root1FP32(width * middle * smallH, k)); }
    } else {
      for (u32 k = 0; k < smallH; ++k) { tab.push_back(root1FancyFP32(smallH * middle, k)); }
      for (u32 k = 0; k < width; ++k)  { tab.push_back(root1FancyFP32(middle * width, k)); }
      for (u32 k = 0; k < smallH; ++k)  { tab.push_back(root1FP32(width * middle * smallH, k)); }
    }
  }
  return tab;
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

// Z31 and GF31 code copied from Yves Gallot's mersenne2 program

// Z/{2^31 - 1}Z: the prime field of order p = 2^31 - 1
class Z31
{
private:
        static const uint32_t _p = (uint32_t(1) << 31) - 1;
        uint32_t _n;    // 0 <= n < p

        static uint32_t _add(const uint32_t a, const uint32_t b)
        {
                const uint32_t t = a + b;
                return t - ((t >= _p) ? _p : 0);
        }

        static uint32_t _sub(const uint32_t a, const uint32_t b)
        {
                const uint32_t t = a - b;
                return t + ((a < b) ? _p : 0);
        }

        static uint32_t _mul(const uint32_t a, const uint32_t b)
        {
                const uint64_t t = a * uint64_t(b);
                return _add(uint32_t(t) & _p, uint32_t(t >> 31));
        }

public:
        Z31() {}
        explicit Z31(const uint32_t n) : _n(n) {}

        uint32_t get() const { return _n; }

        bool operator!=(const Z31 & rhs) const { return (_n != rhs._n); }

        // Z31 neg() const { return Z31((_n == 0) ? 0 : _p - _n); }
        // Z31 half() const { return Z31(((_n % 2 == 0) ? _n : (_n + _p)) / 2); }

        Z31 operator+(const Z31 & rhs) const { return Z31(_add(_n, rhs._n)); }
        Z31 operator-(const Z31 & rhs) const { return Z31(_sub(_n, rhs._n)); }
        Z31 operator*(const Z31 & rhs) const { return Z31(_mul(_n, rhs._n)); }

        Z31 sqr() const { return Z31(_mul(_n, _n)); }
};


// GF((2^31 - 1)^2): the prime field of order p^2, p = 2^31 - 1
class GF31
{
private:
        Z31 _s0, _s1;
        // a primitive root of order 2^32 which is a root of (0, 1).
        static const uint64_t _h_order = uint64_t(1) << 32;
        static const uint32_t _h_0 = 7735u, _h_1 = 748621u;

public:
        GF31() {}
        explicit GF31(const Z31 & s0, const Z31 & s1) : _s0(s0), _s1(s1) {}
        explicit GF31(const uint32_t n0, const uint32_t n1) : _s0(n0), _s1(n1) {}

        const Z31 & s0() const { return _s0; }
        const Z31 & s1() const { return _s1; }

        GF31 operator+(const GF31 & rhs) const { return GF31(_s0 + rhs._s0, _s1 + rhs._s1); }
        GF31 operator-(const GF31 & rhs) const { return GF31(_s0 - rhs._s0, _s1 - rhs._s1); }

        GF31 sqr() const { const Z31 t = _s0 * _s1; return GF31(_s0.sqr() - _s1.sqr(), t + t); }
        GF31 mul(const GF31 & rhs) const { return GF31(_s0 * rhs._s0 - _s1 * rhs._s1, _s1 * rhs._s0 + _s0 * rhs._s1); }

        GF31 pow(const uint64_t e) const
        {
                if (e == 0) return GF31(1u, 0u);
                GF31 r = GF31(1u, 0u), y = *this;
                for (uint64_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
                return r.mul(y);
        }

        static const GF31 root_one(const size_t n) { return GF31(Z31(_h_0), Z31(_h_1)).pow(_h_order / n); }
        static uint8_t log2_root_two(const size_t n) { return uint8_t(((uint64_t(1) << 30) / n) % 31); }
};

// Returns the primitive root of unity of order N, to the power k.
uint2 root1GF31(GF31 root1N, u32 k) {
  GF31 x = root1N.pow(k);
  return { x.s0().get(), x.s1().get() };
}
uint2 root1GF31(u32 N, u32 k) {
  assert(k < N);
  GF31 root1N = GF31::root_one(N);
  return root1GF31(root1N, k);
}

vector<uint2> genSmallTrigGF31(u32 size, u32 radix) {
  u32 WG = size / radix;
  vector<uint2> tab;

  GF31 root1size = GF31::root_one(size);
  for (u32 line = 1; line < radix; ++line) {
    for (u32 col = 0; col < WG; ++col) {
      tab.push_back(root1GF31(root1size, col * line));
    }
  }
  tab.resize(size);
  return tab;
}

// Generate the small trig values for fft_HEIGHT plus optionally trig values used in pairSq.
vector<uint2> genSmallTrigComboGF31(Args *args, u32 width, u32 middle, u32 size, u32 radix, bool tail_single_wide) {
  vector<uint2> tab = genSmallTrigGF31(size, radix);

  u32 tail_trigs = args->value("TAIL_TRIGS31", 0);          // Default is reading all trigs from memory

  // From tailSquareGF31 pre-calculate some or all of these:  GF31 trig = slowTrigGF31(line + H * lowMe, ND / NH * 2);
  u32 height = size;
  GF31 root1wmh = GF31::root_one(width * middle * height);
  if (tail_trigs >= 1) {          // Some trig values in memory, some are computed with a complex multiply.  Best option on a Radeon VII.
    // Output line 0 trig values to be read by every u,v pair of lines
    for (u32 me = 0; me < height / radix; ++me) {
      tab.push_back(root1GF31(root1wmh, width * middle * me));
    }
    // Output the one or two GF31 multipliers to be read by one u,v pair of lines
    for (u32 line = 0; line <= width * middle / 2; ++line) {
      tab.push_back(root1GF31(root1wmh, line));
      if (!tail_single_wide) tab.push_back(root1GF31(root1wmh, line ? width * middle - line : width * middle / 2));
    }
  }
  if (tail_trigs == 0) {          // All trig values read from memory.  Best option for GPUs with great memory performance.
    for (u32 u = 0; u <= width * middle / 2; ++u) {
      for (u32 v = 0; v < (tail_single_wide ? 1 : 2); ++v) {
        u32 line = (v == 0) ? u : (u ? width * middle - u : width * middle / 2);
        for (u32 me = 0; me < height / radix; ++me) {
          tab.push_back(root1GF31(root1wmh, line + width * middle * me));
        }
      }
    }
  }

  return tab;
}

vector<uint2> genMiddleTrigGF31(u32 smallH, u32 middle, u32 width) {
  vector<uint2> tab;
  if (middle == 1) {
    tab.resize(1);
  } else {
    GF31 root1hm = GF31::root_one(smallH * middle);
    for (u32 m = 1; m < middle; ++m) {
      for (u32 k = 0; k < smallH; ++k) { tab.push_back(root1GF31(root1hm, k * m)); }
    }
    GF31 root1mw = GF31::root_one(middle * width);
    for (u32 k = 0; k < width; ++k)  { tab.push_back(root1GF31(root1mw, k)); }
    GF31 root1wmh = GF31::root_one(width * middle * smallH);
    for (u32 k = 0; k < smallH; ++k)  { tab.push_back(root1GF31(root1wmh, k)); }
  }
  return tab;
}


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

// Z61 and GF61 code copied from Yves Gallot's mersenne2 program

// Z/{2^61 - 1}Z: the prime field of order p = 2^61 - 1
class Z61
{
private:
        static const uint64_t _p = (uint64_t(1) << 61) - 1;
        uint64_t _n;    // 0 <= n < p

        static uint64_t _add(const uint64_t a, const uint64_t b)
        {
                const uint64_t t = a + b;
                return t - ((t >= _p) ? _p : 0);
        }

        static uint64_t _sub(const uint64_t a, const uint64_t b)
        {
                const uint64_t t = a - b;
                return t + ((a < b) ? _p : 0);
        }

        static uint64_t _mul(const uint64_t a, const uint64_t b)
        {
                const __uint128_t t = a * __uint128_t(b);
                const uint64_t lo = uint64_t(t), hi = uint64_t(t >> 64);
                const uint64_t lo61 = lo & _p, hi61 = (lo >> 61) | (hi << 3);
                return _add(lo61, hi61);
        }

public:
        Z61() {}
        explicit Z61(const uint64_t n) : _n(n) {}

        uint64_t get() const { return _n; }

        bool operator!=(const Z61 & rhs) const { return (_n != rhs._n); }

        Z61 operator+(const Z61 & rhs) const { return Z61(_add(_n, rhs._n)); }
        Z61 operator-(const Z61 & rhs) const { return Z61(_sub(_n, rhs._n)); }
        Z61 operator*(const Z61 & rhs) const { return Z61(_mul(_n, rhs._n)); }

        Z61 sqr() const { return Z61(_mul(_n, _n)); }
};

// GF((2^61 - 1)^2): the prime field of order p^2, p = 2^61 - 1
class GF61
{
private:
        Z61 _s0, _s1;
        // Primitive root of order 2^62 which is a root of (0, 1).  This root corresponds to 2*pi*i*j/N in FFTs.  PRPLL FFTs use this root.  Thanks, Yves!
        static const uint64_t _h_0 = 264036120304204ull, _h_1 = 4677669021635377ull;
        // Primitive root of order 2^62 which is a root of (0, -1).  This root corresponds to -2*pi*i*j/N in FFTs.
        //static const uint64_t _h_0 = 481139922016222ull, _h_1 = 814659809902011ull;
        static const uint64_t _h_order = uint64_t(1) << 62;

public:
        GF61() {}
        explicit GF61(const Z61 & s0, const Z61 & s1) : _s0(s0), _s1(s1) {}
        explicit GF61(const uint64_t n0, const uint64_t n1) : _s0(n0), _s1(n1) {}

        const Z61 & s0() const { return _s0; }
        const Z61 & s1() const { return _s1; }

        GF61 operator+(const GF61 & rhs) const { return GF61(_s0 + rhs._s0, _s1 + rhs._s1); }
        GF61 operator-(const GF61 & rhs) const { return GF61(_s0 - rhs._s0, _s1 - rhs._s1); }

        GF61 sqr() const { const Z61 t = _s0 * _s1; return GF61(_s0.sqr() - _s1.sqr(), t + t); }
        GF61 mul(const GF61 & rhs) const { return GF61(_s0 * rhs._s0 - _s1 * rhs._s1, _s1 * rhs._s0 + _s0 * rhs._s1); }

        GF61 pow(const uint64_t e) const
        {
                if (e == 0) return GF61(1u, 0u);
                GF61 r = GF61(1u, 0u), y = *this;
                for (uint64_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
                return r.mul(y);
        }

        static const GF61 root_one(const size_t n) { return GF61(Z61(_h_0), Z61(_h_1)).pow(_h_order / n); }
        static uint8_t log2_root_two(const size_t n) { return uint8_t(((uint64_t(1) << 60) / n) % 61); }
};

// Returns the primitive root of unity of order N, to the power k.
ulong2 root1GF61(GF61 root1N, u32 k) {
  GF61 x = root1N.pow(k);
  return { x.s0().get(), x.s1().get() };
}
ulong2 root1GF61(u32 N, u32 k) {
  assert(k < N);
  GF61 root1N = GF61::root_one(N);
  return root1GF61(root1N, k);
}

vector<ulong2> genSmallTrigGF61(u32 size, u32 radix) {
  u32 WG = size / radix;
  vector<ulong2> tab;

  GF61 root1size = GF61::root_one(size);
  for (u32 line = 1; line < radix; ++line) {
    for (u32 col = 0; col < WG; ++col) {
      tab.push_back(root1GF61(root1size, col * line));
    }
  }
  tab.resize(size);
  return tab;
}

// Generate the small trig values for fft_HEIGHT plus optionally trig values used in pairSq.
vector<ulong2> genSmallTrigComboGF61(Args *args, u32 width, u32 middle, u32 size, u32 radix, bool tail_single_wide) {
  vector<ulong2> tab = genSmallTrigGF61(size, radix);

  u32 tail_trigs = args->value("TAIL_TRIGS61", 0);          // Default is reading all trigs from memory

  // From tailSquareGF61 pre-calculate some or all of these:  GF61 trig = slowTrigGF61(line + H * lowMe, ND / NH * 2);
  u32 height = size;
  GF61 root1wmh = GF61::root_one(width * middle * height);
  if (tail_trigs >= 1) {          // Some trig values in memory, some are computed with a complex multiply.  Best option on a Radeon VII.
    // Output line 0 trig values to be read by every u,v pair of lines
    for (u32 me = 0; me < height / radix; ++me) {
      tab.push_back(root1GF61(root1wmh, width * middle * me));
    }
    // Output the one or two GF61 multipliers to be read by one u,v pair of lines
    for (u32 line = 0; line <= width * middle / 2; ++line) {
      tab.push_back(root1GF61(root1wmh, line));
      if (!tail_single_wide) tab.push_back(root1GF61(root1wmh, line ? width * middle - line : width * middle / 2));
    }
  }
  if (tail_trigs == 0) {          // All trig values read from memory.  Best option for GPUs with great memory performance.
    for (u32 u = 0; u <= width * middle / 2; ++u) {
      for (u32 v = 0; v < (tail_single_wide ? 1 : 2); ++v) {
        u32 line = (v == 0) ? u : (u ? width * middle - u : width * middle / 2);
        for (u32 me = 0; me < height / radix; ++me) {
          tab.push_back(root1GF61(root1wmh, line + width * middle * me));
        }
      }
    }
  }

  return tab;
}

vector<ulong2> genMiddleTrigGF61(u32 smallH, u32 middle, u32 width) {
  vector<ulong2> tab;
  if (middle == 1) {
    tab.resize(1);
  } else {
    GF61 root1hm = GF61::root_one(smallH * middle);
    for (u32 m = 1; m < middle; ++m) {
      for (u32 k = 0; k < smallH; ++k) { tab.push_back(root1GF61(root1hm, k * m)); }
    }
    GF61 root1mw = GF61::root_one(middle * width);
    for (u32 k = 0; k < width; ++k)  { tab.push_back(root1GF61(root1mw, k)); }
    GF61 root1wmh = GF61::root_one(width * middle * smallH);
    for (u32 k = 0; k < smallH; ++k)  { tab.push_back(root1GF61(root1wmh, k)); }
  }
  return tab;
}


/**********************************************************/
/*  Build all the needed trig values into one big buffer  */
/**********************************************************/

vector<double2> genSmallTrig(FFTConfig fft, u32 size, u32 radix) {
  vector<double2> tab;
  u32 tabsize;

  if (fft.FFT_FP64) {
    tab = genSmallTrigFP64(size, radix);
    tab.resize(SMALLTRIG_FP64_SIZE(size, 0, 0, 0));
  }

  if (fft.FFT_FP32) {
    vector<float2> tab1 = genSmallTrigFP32(size, radix);
    tab1.resize(SMALLTRIG_FP32_SIZE(size, 0, 0, 0));
    // Append tab1 to tab
    tabsize = tab.size();
    tab.resize(tabsize + tab1.size() / 2);
    memcpy((double *) tab.data() + tabsize * 2, tab1.data(), tab1.size() * 2 * sizeof(float));
  }

  if (fft.NTT_GF31) {
    vector<uint2> tab2 = genSmallTrigGF31(size, radix);
    tab2.resize(SMALLTRIG_GF31_SIZE(size, 0, 0, 0));
    // Append tab2 to tab
    tabsize = tab.size();
    tab.resize(tabsize + tab2.size() / 2);
    memcpy((double *) tab.data() + tabsize * 2, tab2.data(), tab2.size() * 2 * sizeof(uint));
  }

  if (fft.NTT_GF61) {
    vector<ulong2> tab3 = genSmallTrigGF61(size, radix);
    tab3.resize(SMALLTRIG_GF61_SIZE(size, 0, 0, 0));
    // Append tab3 to tab
    tabsize = tab.size();
    tab.resize(tabsize + tab3.size());
    memcpy((double *) tab.data() + tabsize * 2, tab3.data(), tab3.size() * 2 * sizeof(ulong));
  }

  return tab;
}

vector<double2> genSmallTrigCombo(Args *args, FFTConfig fft, u32 width, u32 middle, u32 size, u32 radix, bool tail_single_wide) {
  vector<double2> tab;
  u32 tabsize;

  if (fft.FFT_FP64) {
    tab = genSmallTrigComboFP64(args, width, middle, size, radix, tail_single_wide);
    tab.resize(SMALLTRIGCOMBO_FP64_SIZE(width, middle, size, radix));
  }

  if (fft.FFT_FP32) {
    vector<float2> tab1 = genSmallTrigComboFP32(args, width, middle, size, radix, tail_single_wide);
    tab1.resize(SMALLTRIGCOMBO_FP32_SIZE(width, middle, size, radix));
    // Append tab1 to tab
    tabsize = tab.size();
    tab.resize(tabsize + tab1.size() / 2);
    memcpy((double *) tab.data() + tabsize * 2, tab1.data(), tab1.size() * 2 * sizeof(float));
  }

  if (fft.NTT_GF31) {
    vector<uint2> tab2 = genSmallTrigComboGF31(args, width, middle, size, radix, tail_single_wide);
    tab2.resize(SMALLTRIGCOMBO_GF31_SIZE(width, middle, size, radix));
    // Append tab2 to tab
    tabsize = tab.size();
    tab.resize(tabsize + tab2.size() / 2);
    memcpy((double *) tab.data() + tabsize * 2, tab2.data(), tab2.size() * 2 * sizeof(uint));
  }

  if (fft.NTT_GF61) {
    vector<ulong2> tab3 = genSmallTrigComboGF61(args, width, middle, size, radix, tail_single_wide);
    tab3.resize(SMALLTRIGCOMBO_GF61_SIZE(width, middle, size, radix));
    // Append tab3 to tab
    tabsize = tab.size();
    tab.resize(tabsize + tab3.size());
    memcpy((double *) tab.data() + tabsize * 2, tab3.data(), tab3.size() * 2 * sizeof(ulong));
  }

  return tab;
}

vector<double2> genMiddleTrig(FFTConfig fft, u32 smallH, u32 middle, u32 width) {
  vector<double2> tab;
  u32 tabsize;

  if (fft.FFT_FP64) {
    tab = genMiddleTrigFP64(smallH, middle, width);
    tab.resize(MIDDLETRIG_FP64_SIZE(width, middle, smallH));
  }

  if (fft.FFT_FP32) {
    vector<float2> tab1 = genMiddleTrigFP32(smallH, middle, width);
    tab1.resize(MIDDLETRIG_FP32_SIZE(width, middle, smallH));
    // Append tab1 to tab
    tabsize = tab.size();
    tab.resize(tabsize + tab1.size() / 2);
    memcpy((double *) tab.data() + tabsize * 2, tab1.data(), tab1.size() * 2 * sizeof(float));
  }

  if (fft.NTT_GF31) {
    vector<uint2> tab2 = genMiddleTrigGF31(smallH, middle, width);
    tab2.resize(MIDDLETRIG_GF31_SIZE(width, middle, smallH));
    // Append tab2 to tab
    tabsize = tab.size();
    tab.resize(tabsize + tab2.size() / 2);
    memcpy((double *) tab.data() + tabsize * 2, tab2.data(), tab2.size() * 2 * sizeof(uint));
  }

  if (fft.NTT_GF61) {
    vector<ulong2> tab3 = genMiddleTrigGF61(smallH, middle, width);
    tab3.resize(MIDDLETRIG_GF61_SIZE(width, middle, smallH));
    // Append tab3 to tab
    tabsize = tab.size();
    tab.resize(tabsize + tab3.size());
    memcpy((double *) tab.data() + tabsize * 2, tab3.data(), tab3.size() * 2 * sizeof(ulong));
  }

  return tab;
}


/********************************************************/
/*        Code to manage a cache of trigBuffers         */
/********************************************************/

#define make_key_part(b,tt,b31,tt31,b32,tt32,b61,tt61,tk) ((((((((b+tt) << 2) + b31+tt31) << 2) + b32+tt32) << 2) + b61+tt61) << 2) + tk

TrigBufCache::~TrigBufCache() = default;

TrigPtr TrigBufCache::smallTrig(Args *args, FFTConfig fft, u32 width, u32 nW, u32 middle, u32 height, u32 nH, bool tail_single_wide) {
  lock_guard lock{mut};
  auto& m = small;
  TrigPtr p{};

  u32 tail_trigs = args->value("TAIL_TRIGS", 2);                 // Default is calculating FP64 trigs from scratch, no memory accesses
  u32 tail_trigs31 = args->value("TAIL_TRIGS31", 2);             // Default is reading GF31 trigs from memory
  u32 tail_trigs32 = args->value("TAIL_TRIGS32", 2);             // Default is calculating FP32 trigs from scratch, no memory accesses
  u32 tail_trigs61 = args->value("TAIL_TRIGS61", 2);             // Default is reading GF61 trigs from memory
  u32 key_part = make_key_part(fft.FFT_FP64, tail_trigs, fft.NTT_GF31, tail_trigs31, fft.FFT_FP32, tail_trigs32, fft.NTT_GF61, tail_trigs61, tail_single_wide);

  // See if there is an existing smallTrigCombo that we can return (using only a subset of the data)
  // In theory, we could match any smallTrigCombo where width matches.  However, SMALLTRIG_GF31_SIZE wouldn't be able to figure out the size.
  // In practice, those cases will likely never arise.
  if (width == height && nW == nH) {
    decay_t<decltype(m)>::key_type key{height, nH, width, middle, key_part};
    auto it = m.find(key);
    if (it != m.end() && (p = it->second.lock())) return p;
  }

  // See if there is an existing non-combo smallTrig that we can return
  decay_t<decltype(m)>::key_type key{width, nW, 0, 0, key_part}; 
  auto it = m.find(key);
  if (it != m.end() && (p = it->second.lock())) return p;

  // Create a new non-combo
  p = make_shared<TrigBuf>(context, genSmallTrig(fft, width, nW));
  m[key] = p;
  smallCache.add(p);
  return p;
}

TrigPtr TrigBufCache::smallTrigCombo(Args *args, FFTConfig fft, u32 width, u32 middle, u32 height, u32 nH, bool tail_single_wide) {
  u32 tail_trigs = args->value("TAIL_TRIGS", 2);                 // Default is calculating FP64 trigs from scratch, no memory accesses
  u32 tail_trigs31 = args->value("TAIL_TRIGS31", 2);             // Default is reading GF31 trigs from memory
  u32 tail_trigs32 = args->value("TAIL_TRIGS32", 2);             // Default is calculating FP32 trigs from scratch, no memory accesses
  u32 tail_trigs61 = args->value("TAIL_TRIGS61", 2);             // Default is reading GF61 trigs from memory
  u32 key_part = make_key_part(fft.FFT_FP64, tail_trigs, fft.NTT_GF31, tail_trigs31, fft.FFT_FP32, tail_trigs32, fft.NTT_GF61, tail_trigs61, tail_single_wide);

  // If there are no pre-computed trig values we might be able to share this trig table with fft_WIDTH
  if (((tail_trigs == 2 && fft.FFT_FP64) || (tail_trigs32 == 2 && fft.FFT_FP32)) && !fft.NTT_GF31 && !fft.NTT_GF61)
    return smallTrig(args, fft, height, nH, middle, height, nH, tail_single_wide);

  lock_guard lock{mut};
  auto& m = small;
  decay_t<decltype(m)>::key_type key{height, nH, width, middle, key_part};

  TrigPtr p{};
  auto it = m.find(key);
  if (it == m.end() || !(p = it->second.lock())) {
    p = make_shared<TrigBuf>(context, genSmallTrigCombo(args, fft, width, middle, height, nH, tail_single_wide));
    m[key] = p;
    smallCache.add(p);
  }
  return p;
}

TrigPtr TrigBufCache::middleTrig(Args *args, FFTConfig fft, u32 SMALL_H, u32 MIDDLE, u32 width) {
  lock_guard lock{mut};
  auto& m = middle;
  u32 key_part = make_key_part(fft.FFT_FP64, 0, fft.NTT_GF31, 0, fft.FFT_FP32, 0, fft.NTT_GF61, 0, 0);
  decay_t<decltype(m)>::key_type key{SMALL_H, MIDDLE, width, key_part};

  TrigPtr p{};
  auto it = m.find(key);
  if (it == m.end() || !(p = it->second.lock())) {
    p = make_shared<TrigBuf>(context, genMiddleTrig(fft, SMALL_H, MIDDLE, width));
    m[key] = p;
    middleCache.add(p);
  }
  return p;
}

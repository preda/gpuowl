// Copyright Mihai Preda

#include "TrigBufCache.h"

#define SAVE_ONE_MORE_WIDTH_MUL  0      // I want to make saving the only option -- but rocm optimizer is inexplicably making it slower in carryfused
#define SAVE_ONE_MORE_HEIGHT_MUL 1      // In tailSquar this is the fastest option

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

namespace {
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

vector<double2> genSmallTrig(u32 size, u32 radix) {
  if (LOG_TRIG_ALLOC) { log("genSmallTrig(%u, %u)\n", size, radix); }
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
vector<double2> genSmallTrigCombo(u32 width, u32 middle, u32 size, u32 radix, bool tail_single_wide, u32 tail_trigs) {
  if (LOG_TRIG_ALLOC) { log("genSmallTrigCombo(%u, %u)\n", size, radix); }

  vector<double2> tab = genSmallTrig(size, radix);

  // From tailSquare pre-calculate some or all of these:  T2 trig = slowTrig_N(line + H * lowMe, ND / NH * 2);
  if (tail_trigs == 1) {          // Some trig values in memory, some are computed with a complex multiply.  Best option on a Radeon VII.
    u32 height = size;
    // Output line 0 trig values to be read by every u,v pair of lines
    for (u32 me = 0; me < height / radix; ++me) {
      tab.push_back(root1(width * middle * height, width * middle * me));
    }
    // Output the one or two T2 multipliers to be read by one u,v pair of lines
    for (u32 line = 0; line < width * middle / 2; ++line) {
      tab.push_back(root1Fancy(width * middle * height, line));
      if (!tail_single_wide) tab.push_back(root1Fancy(width * middle * height, width * middle - line));
    }
  }
  if (tail_trigs == 0) {          // All trig values read from memory.  Best option for GPUs with lousy DP performance.
    u32 height = size;
    for (u32 u = 0; u < width * middle / 2; ++u) {
      for (u32 v = 0; v < (tail_single_wide ? 1 : 2); ++v) {
        u32 line = (v == 0) ? u : width * middle - u;
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

vector<double2> genMiddleTrig(u32 smallH, u32 middle, u32 width) {
  if (LOG_TRIG_ALLOC) { log("genMiddleTrig(%u, %u, %u)\n", smallH, middle, width); }
  vector<double2> tab;
  if (middle == 1) {
    tab.resize(1);
  } else {
    if (middle < SHARP_MIDDLE) {
      for (u32 k = 0; k < smallH; ++k) { tab.push_back(root1(smallH * middle, k)); }
      for (u32 k = 0; k < width; ++k)  { tab.push_back(root1(middle * width, k)); }
    } else {
      for (u32 k = 0; k < smallH; ++k) { tab.push_back(root1Fancy(smallH * middle, k)); }
      for (u32 k = 0; k < width; ++k)  { tab.push_back(root1Fancy(middle * width, k)); }
    }
  }
  return tab;
}

} // namespace

TrigBufCache::~TrigBufCache() = default;

TrigPtr TrigBufCache::smallTrig(u32 W, u32 nW) {
  lock_guard lock{mut};
  auto& m = small;
  decay_t<decltype(m)>::key_type key{W, nW, 0, 0};

  TrigPtr p{};
  auto it = m.find(key);
  if (it == m.end() || !(p = it->second.lock())) {
    p = make_shared<TrigBuf>(context, genSmallTrig(W, nW));
    m[key] = p;
    smallCache.add(p);
  }
  return p;
}

TrigPtr TrigBufCache::smallTrigCombo(u32 width, u32 middle, u32 W, u32 nW, u32 variant, bool tail_single_wide, u32 tail_trigs) {
  if (tail_trigs == 2)             // No pre-computed trig values.  We might be able to share this trig table with fft_WIDTH
    return smallTrig(W, nW);

  lock_guard lock{mut};
  auto& m = small;
  decay_t<decltype(m)>::key_type key1{W, nW, width, middle};
  // We write the "combo" under two keys, so it can also be retrieved as non-combo by smallTrig()
  decay_t<decltype(m)>::key_type key2{W, nW, 0, 0};

  TrigPtr p{};
  auto it = m.find(key1);
  if (it == m.end() || !(p = it->second.lock())) {
    p = make_shared<TrigBuf>(context, genSmallTrigCombo(width, middle, W, nW, tail_single_wide, tail_trigs));
    m[key1] = p;
    m[key2] = p;
    smallCache.add(p);
  }
  return p;
}

TrigPtr TrigBufCache::middleTrig(u32 SMALL_H, u32 MIDDLE, u32 width) {
  lock_guard lock{mut};
  auto& m = middle;
  decay_t<decltype(m)>::key_type key{SMALL_H, MIDDLE, width};

  TrigPtr p{};
  auto it = m.find(key);
  if (it == m.end() || !(p = it->second.lock())) {
    p = make_shared<TrigBuf>(context, genMiddleTrig(SMALL_H, MIDDLE, width));
    m[key] = p;
    middleCache.add(p);
  }
  return p;
}

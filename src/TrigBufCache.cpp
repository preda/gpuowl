// Copyright Mihai Preda

#include "TrigBufCache.h"

// Trial balloon.  I'm hoping we can create one #define that indicates a GPU's preference for doing extra DP work in exchange for
// less memory accesses.  I can envision a Titan V (slow memory, 1:2 SP:DP ratio) setting this value to a high value, whereas a
// consumer grade nVidia GPU with 1:32 or 1:64 SP:DP ratio would set this to zero.
#define PREFER_DP_TO_MEM        2               // Excellent DP GPU such as Titan V or Radeon VII Pro.
//#define PREFER_DP_TO_MEM      1               // Good DP GPU.  Tuned for Radeon VII.
//#define PREFER_DP_TO_MEM      0               // Poor DP GPU.  A typical consumer grade GPU.

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

namespace {
static const constexpr bool LOG_TRIG_ALLOC = false;

vector<double2> genSmallTrig(u32 size, u32 radix) {
  if (LOG_TRIG_ALLOC) { log("genSmallTrig(%u, %u)\n", size, radix); }

  vector<double2> tab;
#if 1
  for (u32 line = 1; line < radix; ++line) {
    for (u32 col = 0; col < size / radix; ++col) {
      tab.push_back(radix / line >= 8 ? root1Fancy(size, col * line) : root1(size, col * line));
    }
  }
  tab.resize(size);
#else
  tab.resize(size);
  auto *p = tab.data() + radix;
  for (u32 w = radix; w < size; w *= radix) { p = smallTrigBlock(w, std::min(radix, size / w), p); }
  assert(p - tab.data() == size);
#endif

  return tab;
}

// Generate the small trig values for fft_HEIGHT plus optionally trig values used in pairSq.
vector<double2> genSmallTrigCombo(u32 width, u32 middle, u32 size, u32 radix) {
  if (LOG_TRIG_ALLOC) { log("genSmallTrigCombo(%u, %u)\n", size, radix); }

  vector<double2> tab;
  for (u32 line = 1; line < radix; ++line) {
    for (u32 col = 0; col < size / radix; ++col) {
      tab.push_back(radix / line >= 8 ? root1Fancy(size, col * line) : root1(size, col * line));
    }
  }
  // From tailSquare pre-calculate some or all of these:  T2 trig = slowTrig_N(line + H * lowMe, ND / NH * 2);
#if PREFER_DP_TO_MEM == 2             // No pre-computed trig values
#elif PREFER_DP_TO_MEM == 1           // best option on a Radeon VII.
  u32 height = size;
  // Output line 0 trig values to be read by every u,v pair of lines
  for (u32 me = 0; me < height / radix; ++me) {
    tab.push_back(root1(width * middle * height, width * middle * me));
  }
  // Output the two T2 multipliers to be read by one u,v pair of lines
  for (u32 line = 0; line < width * middle / 2; ++line) {
    tab.push_back(root1Fancy(width * middle * height, line));
    tab.push_back(root1Fancy(width * middle * height, width * middle - line));
  }
#else
  u32 height = size;
  for (u32 u = 0; u < width * middle / 2; ++u) {
    for (u32 v = 0; v < 2; ++v) {
      u32 line = (v == 0) ? u : width * middle - u;
      for (u32 me = 0; me < height / radix; ++me) {
        tab.push_back(root1(width * middle * height, line + width * middle * me));
      }
    }
  }
#endif

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
  decay_t<decltype(m)>::key_type key{W, nW};

  TrigPtr p{};
  auto it = m.find(key);
  if (it == m.end() || !(p = it->second.lock())) {
    p = make_shared<TrigBuf>(context, genSmallTrig(W, nW));
    m[key] = p;
    smallCache.add(p);
  }
  return p;
}

TrigPtr TrigBufCache::smallTrigCombo(u32 width, u32 middle, u32 W, u32 nW) {
  lock_guard lock{mut};
  auto& m = small;
#if PREFER_DP_TO_MEM == 2             // No pre-computed trig values
  decay_t<decltype(m)>::key_type key{W, nW};
#else
  // Hack so that width 512 and height 512 don't share the same buffer.  Width could share the height buffer since it is a subset of the combo height buffer.
  decay_t<decltype(m)>::key_type key{W, nW+1};
#endif

  TrigPtr p{};
  auto it = m.find(key);
  if (it == m.end() || !(p = it->second.lock())) {
    p = make_shared<TrigBuf>(context, genSmallTrigCombo(width, middle, W, nW));
    m[key] = p;
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

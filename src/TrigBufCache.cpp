// Copyright Mihai Preda

#include "TrigBufCache.h"

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static_assert(sizeof(double2) == 16, "size double2");
static_assert(sizeof(long double) > sizeof(double), "long double offers extended precision");

namespace {
static const constexpr bool LOG_TRIG_ALLOC = false;

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
    return {-s, -c};
  } else {
    assert(!(N&7));
    assert(k <= N/8);
    N /= 2;

    double angle = - M_PIl * k / N;
    return {cos(angle), sin(angle)};
  }
}

// For small angles, return "fancy" cos - 1 for increased precision
double2 root1Fancy(u32 N, u32 k) {
  assert(!(N&7));
  assert(k < N);
  assert(k < N/4);

  double angle = - M_PIl * k / (N / 2);
  return {double(cosl(angle) - 1), sin(angle)};
}

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

template<typename T>
vector<pair<T, T>> makeTrig(u32 n, vector<pair<T,T>> tab = {}) {
  assert(n % 8 == 0);
  tab.reserve(tab.size() + n/8 + 1);
  for (u32 k = 0; k <= n/8; ++k) { tab.push_back(root1(n, k)); }
  return tab;
}

template<typename T>
vector<pair<T, T>> makeTinyTrig(u32 W, u32 hN, vector<pair<T, T>> tab = {}) {
  tab.reserve(tab.size() + W/2 + 1);
  for (u32 k = 0; k <= W/2; ++k) {
    tab.push_back(root1Fancy(hN, k));
  }
  return tab;
}

vector<double2> makeSquareTrig(u32 hN, u32 nH, u32 smallH) {
  if (LOG_TRIG_ALLOC) { log("makeSquareTrig(%u, %u, %u)\n", hN, nH, smallH); }
  vector<double2> ret;

  assert(hN % (smallH * 2) == 0);
  u32 nGroups = hN / (smallH * 2);

  for (u32 me = 0; me < smallH / nH; ++me) {
    ret.push_back(root1(hN, me * (hN / smallH)));
  }

  for (u32 g = 0; g < nGroups; ++g) {
    ret.push_back(root1Fancy(hN, g == 0 ? nGroups : g));
  }

#if 0 // Old implem
  for (u32 i = 0; i < nGroups; ++i) {
    for (u32 me = 0; me < smallH / nH; ++me) {
      ret.push_back(root1(hN, i + me * (hN / smallH)));
    }
  }
  assert(ret.size() == (hN / (2 * nH)));
#endif
  return ret;
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

TrigPtr TrigBufCache::trigBHW(u32 W, u32 hN, u32 BIG_H) {
  lock_guard lock{mut};
  auto& m = bhw;
  decay_t<decltype(m)>::key_type key{W, hN, BIG_H};

  TrigPtr p{};
  auto it = m.find(key);
  if (it == m.end() || !(p = it->second.lock())) {
    if (LOG_TRIG_ALLOC) { log("trigBHW(%u, %u, %u)\n", W, hN, BIG_H); }
    p = make_shared<TrigBuf>(context, makeTinyTrig(W, hN, makeTrig<double>(BIG_H)));
    m[key] = p;
    bhwCache.add(p);
  }
  return p;
}

TrigPtr TrigBufCache::trigSquare(u32 hN, u32 nH, u32 SMALL_H) {
  lock_guard lock{mut};
  auto& m = square;
  decay_t<decltype(m)>::key_type key{hN, nH, SMALL_H};

  TrigPtr p{};
  auto it = m.find(key);
  if (it == m.end() || !(p = it->second.lock())) {
    p = make_shared<TrigBuf>(context, makeSquareTrig(hN, nH, SMALL_H));
    m[key] = p;
    squareCache.add(p);
  }
  return p;
}

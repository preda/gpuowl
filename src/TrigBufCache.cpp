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

double2 root1Fancy(u32 N, u32 k) {
  assert(!(N&7));
  assert(k < N);
  if (k <= N/8) {
    double angle = - M_PIl * k / (N / 2);
    return {double(cosl(angle) - 1), sin(angle)};
  }
  auto [c, s] = root1(N, k);
  return {c-1, s};
}

[[maybe_unused]] double2 *smallTrigBlock(u32 W, u32 H, double2 *p) {
  for (u32 line = 1; line < H; ++line) {
    for (u32 col = 0; col < W; ++col) {
      *p++ = root1(W * H, line * col);
    }
  }
  return p;
}

double2 *smallTrigBlockTransp(u32 W, u32 H, double2 *p) {
  for (u32 col = 0; col < W; ++col) {
    for (u32 line = 1; line < H; ++line) {
      *p++ = root1Fancy(W * H, line * col);
    }
  }
  return p;
}

vector<double2> genSmallTrig(u32 size, u32 radix) {
  vector<double2> tab;

#if 1
  for (u32 line = 1; line < radix; ++line) {
    for (u32 col = 0; col < size / radix; ++col) {
      tab.push_back(root1(size, col * line));
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

vector<double2> genMiddleTrig(u32 smallH, u32 middle) {
  vector<double2> tab;
  if (middle == 1) {
    tab.resize(1);
  } else {
    u32 size = smallH * (middle - 1);
    tab.resize(size);
    [[maybe_unused]] auto *p = smallTrigBlockTransp(smallH, middle, tab.data());
    assert(p - tab.data() == size);
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
    auto[c, s] = root1(hN, k);
    tab.push_back({c - 1, s});
  }
  return tab;
}

vector<double2> makeSquareTrig(u32 hN, u32 nH, u32 smallH) {
  vector<double2> ret;

  assert(hN % (smallH * 2) == 0);
  u32 nGroups = hN / (smallH * 2);
  for (u32 i = 0; i < nGroups; ++i) {
    for (u32 me = 0; me < smallH / nH; ++me) {
      ret.push_back(root1(hN, i + me * (hN / smallH)));
    }
  }
  assert(ret.size() == (hN / (2 * nH)));
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
  }
  lastSmall = p;
  return p;
}

TrigPtr TrigBufCache::middleTrig(u32 SMALL_H, u32 nH) {
  lock_guard lock{mut};
  auto& m = middle;
  decay_t<decltype(m)>::key_type key{SMALL_H, nH};

  TrigPtr p{};
  auto it = m.find(key);
  if (it == m.end() || !(p = it->second.lock())) {
    p = make_shared<TrigBuf>(context, genMiddleTrig(SMALL_H, nH));
    m[key] = p;
  }
  lastMiddle = p;
  return p;
}

TrigPtr TrigBufCache::trigBHW(u32 W, u32 hN, u32 BIG_H) {
  lock_guard lock{mut};
  auto& m = bhw;
  decay_t<decltype(m)>::key_type key{W, hN, BIG_H};

  TrigPtr p{};
  auto it = m.find(key);
  if (it == m.end() || !(p = it->second.lock())) {
    p = make_shared<TrigBuf>(context, makeTinyTrig(W, hN, makeTrig<double>(BIG_H)));
    m[key] = p;
  }
  lastBHW = p;
  return p;
}

TrigPtr TrigBufCache::trig2SH(u32 SMALL_H) {
  lock_guard lock{mut};
  auto& m = sh;
  decay_t<decltype(m)>::key_type key{SMALL_H};

  TrigPtr p{};
  auto it = m.find(key);
  if (it == m.end() || !(p = it->second.lock())) {
    p = make_shared<TrigBuf>(context, makeTrig<double>(2 * SMALL_H));
    m[key] = p;
  }
  last2SH = p;
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
  }
  lastSquare = p;
  return p;
}

// Copyright Mihai Preda

#pragma once

#include "common.h"

template <typename H>
class Hash {
  H h;
  
public:
  template <typename... Ts>
  static array<u64, 4> hash(Ts... data) {
    Hash hash;
    (hash.update(data),...);
    return std::move(hash).finish();
  }

  void update(u32 x) { h.update(&x, sizeof(x)); }
  void update(u64 x) { h.update(&x, sizeof(x)); }
  
  template<typename T>
  void update(const vector<T>& v) { h.update(v.data(), v.size() * sizeof(T)); }

  template<typename T, std::size_t N>
  void update(const array<T, N>& v) { h.update(v.data(), N * sizeof(T)); }

  void update(const string& s) {h.update(s.c_str(), s.size()); }
  
  array<u64, 4> finish() && { return std::move(h).finish(); }
};

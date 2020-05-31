// Copyright Mihai Preda

#pragma once

#include "common.h"
#include <initializer_list>

struct HashData {
  const unsigned char* begin;
  const size_t size;
  union {
    u32 data32;
    u64 data64;
  };
    
  HashData(const unsigned char* ptr, size_t size) : begin{ptr}, size{size} {}
  HashData(const void* ptr, size_t size) : HashData{static_cast<const unsigned char*>(ptr), size} {}
  
  HashData(u32 x) : HashData{&data32, sizeof(x)} { data32 = x; }
  HashData(u64 x) : HashData{&data64, sizeof(x)} { data64 = x; }
    
  template<typename T>
  HashData(const std::vector<T>& v) : HashData{v.data(), v.size() * sizeof(T)} {}
  
  template<typename T, std::size_t N>
  HashData(const array<T, N>& v) : HashData(v.data(), N * sizeof(T)) {}
};


template <typename H>
class Hash {  
public:
  static array<u64, 4> hash(std::initializer_list<HashData> datas) {
    H h;
    for (HashData data : datas) { h.update(data.begin, data.size); }
    return std::move(h).finish();
  }
};

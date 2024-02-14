// Copyright (C) Mihai Preda.

#pragma once

#include "log.h"
#include <atomic>
#include <new>

using namespace std::string_literals;

class AllocTrac {
  static std::atomic<size_t> totalAlloc;  
  static size_t maxAlloc;
  
  size_t size{};
  
public:
  AllocTrac() = default;
  explicit AllocTrac(size_t size) : size(size) {
    if (size) {
      if (totalAlloc + size >= maxAlloc) {
        log("Reached GPU maxAlloc limit %.1f GB\n", float(maxAlloc) / (1024 * 1024 * 1024));
        throw std::bad_alloc();
      }
      totalAlloc += size;
      // log("alloc %lu total %lu limit %lu\n", size, size_t(totalAlloc), maxAlloc);
    }
  }
  ~AllocTrac() {
    if (size) {
      totalAlloc -= size;
      // log("release %lu total %lu limit %lu\n", size, size_t(totalAlloc), maxAlloc);
    }
  }

  AllocTrac(const AllocTrac&) = delete;
  void operator=(const AllocTrac&) = delete;

  AllocTrac(AllocTrac&& rhs) : size(rhs.size) { rhs.size = 0; }
  AllocTrac& operator=(AllocTrac&& rhs) {
    AllocTrac tmp{std::move(rhs)};
    swap(*this, tmp);
    return *this;
  }

  friend void swap(AllocTrac& a, AllocTrac& b) noexcept {
    using std::swap;
    swap(a.size, b.size);
  }

  static void setMaxAlloc(size_t m) { maxAlloc = m; }
  static size_t totalAllocBytes() { return totalAlloc; }
  static size_t availableBytes() { return maxAlloc - totalAlloc; }
};

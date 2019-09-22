#pragma once

#include <atomic>
#include <new>
#include <string>

using namespace std::string_literals;

class gpu_bad_alloc : public std::bad_alloc {
  std::string w;
  
public:
  gpu_bad_alloc(const std::string& w) : w(w) {}
  gpu_bad_alloc(size_t size) : gpu_bad_alloc("GPU size "s + std::to_string(size)) {}

  const char *what() const noexcept override { return w.c_str(); }
};

class AllocTrac {
  static std::atomic<size_t> totalAlloc;  
  static size_t maxAlloc;
  
  size_t size{};
  
public:
  AllocTrac() = default;
  AllocTrac(size_t size) : size(size) {
    if (size) {
      if (totalAlloc + size >= maxAlloc) { throw gpu_bad_alloc("Reached GPU maxAlloc limit " + std::to_string(maxAlloc >> 20) + " MB"); }
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
  
  static size_t totalAllocBytes() { return totalAlloc; }
  static void setMaxAlloc(size_t m) { maxAlloc = m; }
};

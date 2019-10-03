// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"
#include <cstdio>
#include <cstdarg>
#include <memory>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace std {
  template<> struct default_delete<FILE> {
    void operator()(FILE *f) { if (f != nullptr) { fclose(f); } }
  };
}

class File {
  std::unique_ptr<FILE> ptr;
  std::string name;

  File(std::unique_ptr<FILE>&& ptr, std::string_view name) : ptr{std::move(ptr)}, name{name} {}

  static File open(const fs::path &name, const char *mode, bool doLog) {
    std::string sname{name.string()};
    std::unique_ptr<FILE> f{fopen(sname.c_str(), mode)};
    if (!f && doLog) {
      log("Can't open '%s' (mode '%s')\n", name.c_str(), mode);
      throw(fs::filesystem_error("can't open file"s, name, {}));
    }
    return {std::move(f), sname};
  }
  
public:
  static File openRead(const fs::path& name, bool doThrow = false) { return open(name, "rb", doThrow); }
  static File openWrite(const fs::path &name) { return open(name, "wb", true); }
  static File openAppend(const fs::path &name) { return open(name, "ab", true); }
  static File openReadAppend(const fs::path &name) { return open(name, "a+b", true); }

  File(FILE* ptr, std::string_view name) : ptr{ptr}, name{name} {}
  
  template<typename T>
  void write(const vector<T>& v) {
    if (!fwrite(v.data(), v.size() * sizeof(T), 1, get())) { throw(std::ios_base::failure((name + "can't write data ").c_str())); }
  }
  
  int printf(const char *fmt, ...) __attribute__((format(printf, 2, 3))) {
    va_list va;
    va_start(va, fmt);
    int ret = vfprintf(ptr.get(), fmt, va);
    va_end(va);
    return ret;
  }

  operator bool() const { return bool(ptr); }
  FILE* get() { return ptr.get(); }
};

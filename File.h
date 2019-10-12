// Copyright (C) Mihai Preda.

#pragma once

#include "common.h"
#include <cstdio>
#include <cstdarg>
#include <cassert>
#include <memory>
#include <filesystem>
#include <vector>
#include <string>

namespace fs = std::filesystem;

namespace std {
  template<> struct default_delete<FILE> {
    void operator()(FILE *f) { if (f != nullptr) { fclose(f); } }
  };
}

class File {
  std::unique_ptr<FILE> ptr;

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
  static fs::path fileName(u32 E, const string& suffix = "", const string& extension = "owl") {
    string sE = to_string(E);
    auto baseDir = fs::current_path() / sE;
    if (!fs::exists(baseDir)) { fs::create_directory(baseDir); }
    return baseDir / (sE + suffix + '.' + extension);
  }

  static File openRead(const fs::path& name, bool doThrow = false) { return open(name, "rb", doThrow); }
  static File openWrite(const fs::path &name) { return open(name, "wb", true); }
  static File openAppend(const fs::path &name) { return open(name, "ab", true); }
  static File openReadAppend(const fs::path &name) { return open(name, "a+b", true); }
  static File openReadAppend(u32 E, const string& extension) { return openReadAppend(fileName(E, "", extension)); }

  File(FILE* ptr, std::string_view name) : ptr{ptr}, name{name} {}

  const std::string name;
  
  template<typename T>
  void write(const vector<T>& v) {
    if (!fwrite(v.data(), v.size() * sizeof(T), 1, get())) { throw(std::ios_base::failure((name + ": can't write data").c_str())); }
  }

  void flush() { fflush(get()); }
  
  int printf(const char *fmt, ...) __attribute__((format(printf, 2, 3))) {
    va_list va;
    va_start(va, fmt);
    int ret = vfprintf(ptr.get(), fmt, va);
    va_end(va);
    return ret;
  }

  operator bool() const { return bool(ptr); }
  FILE* get() const { return ptr.get(); }

  long ftell() const {
    long pos = ::ftell(get());
    assert(pos >= 0);
    return pos;
  }

  void seek(long pos) {
    int err = fseek(get(), pos, SEEK_SET);
    assert(!err);
  }

  long seekEnd() {
    int err = fseek(get(), 0, SEEK_END);
    assert(!err);
    return ftell();
  }
  
  long size() {
    long savePos = ftell();
    long retSize = seekEnd();
    seek(savePos);
    return retSize;
  }

  bool empty() { return size() == 0; }

  std::string readLine() {
    char buf[256];
    return fgets(buf, sizeof(buf), get()) ? buf : "";
  }

  template<typename T>
  std::vector<T> read(u32 nWords) {
    vector<T> ret;
    ret.resize(nWords);
    if (!fread(ret.data(), nWords * sizeof(T), 1, get())) {
      throw(std::ios_base::failure(name + ": can't read"));
    }
    return ret;
  }
};

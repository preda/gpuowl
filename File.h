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
#include <optional>

namespace fs = std::filesystem;

namespace std {
  template<> struct default_delete<FILE> {
    void operator()(FILE *f) { if (f != nullptr) { fclose(f); } }
  };
}

class File {
  std::unique_ptr<FILE> ptr;

  File() = default;
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
  class It {
  public:
    It(File& file) : file{&file}, line{file.maybeReadLine()} {}
    It() = default;

    bool operator==(const It& rhs) const { return !line && !rhs.line; }
    bool operator!=(const It& rhs) const { return !(*this == rhs); }
    
    It& operator++() {
      line = file->maybeReadLine();
      return *this;
    }
    
    string operator*() { return *line; }

  private:
    File *file{};
    optional<string> line;
    // optional<string> next;
  };

  It begin() { return It{*this}; }
  It end() { return It{}; }
  
  static File openRead(const fs::path& name, bool doThrow = false) { return open(name, "rb", doThrow); }
  static File openWrite(const fs::path &name) { return open(name, "wb", true); }
  static File openAppend(const fs::path &name) { return open(name, "ab", true); }
  static void append(const fs::path& name, std::string_view s) { File::openAppend(name).write(s); }
  
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

  void write(string_view s) {
    if (fwrite(s.data(), s.size(), 1, ptr.get()) != 1) {
      throw fs::filesystem_error("can't write to file"s, name, {});
    }
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
    char buf[512];
    bool ok = fgets(buf, sizeof(buf), get());
    return ok ? buf : "";
  }

  std::optional<std::string> maybeReadLine() {
    char buf[512];
    bool ok = fgets(buf, sizeof(buf), get());
    std::optional<string> ret;
    if (ok) { ret = buf; }
    return ret;
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

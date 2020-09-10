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

class File {
  FILE* f = nullptr;

  File(const fs::path &path, const string& mode, bool doLog) : name{path.string()} {
    f = fopen(name.c_str(), mode.c_str());
    if (!f && doLog) {
      log("Can't open '%s' (mode '%s')\n", name.c_str(), mode.c_str());
      throw(fs::filesystem_error("can't open file"s, path, {}));
    }
  }
    
public:
  File(FILE* f, const string& name) : f{f}, name{name} {}
  File(File&& other) : f{other.f}, name{other.name} { other.f = nullptr; }
  
  File(const File& other) = delete;
  File& operator=(const File& other) = delete;
  File& operator=(File&& other) = delete;

  ~File() {
    if (f != nullptr) { fclose(f); }    
  }
  
  class It {
  public:
    It(File& file) : file{&file}, line{file ? file.maybeReadLine() : nullopt} {}
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
  };

  It begin() { return It{*this}; }
  It end() { return It{}; }
  
  static File openRead(const fs::path& name, bool doThrow = false) { return File{name, "rb", doThrow}; }
  static File openWrite(const fs::path &name) { return File{name, "wb", true}; }
  static File openAppend(const fs::path &name) { return File{name, "ab", true}; }
  static void append(const fs::path& name, std::string_view s) { File::openAppend(name).write(s); }

  const std::string name;
  
  template<typename T>
  void write(const vector<T>& v) { write(v.data(), v.size() * sizeof(T)); }

  void write(const void* data, u32 nBytes) {
    if (!fwrite(data, nBytes, 1, get())) { throw(std::ios_base::failure((name + ": can't write data").c_str())); }
  }
  
  void flush() { fflush(get()); }
  
  int printf(const char *fmt, ...) __attribute__((format(printf, 2, 3))) {
    va_list va;
    va_start(va, fmt);
    int ret = vfprintf(f, fmt, va);
    va_end(va);
    return ret;
  }

  int scanf(const char *fmt, ...) __attribute__((format(scanf, 2, 3))) {
    va_list va;
    va_start(va, fmt);
    int ret = vfscanf(f, fmt, va);
    va_end(va);
    return ret;
  }
  
  void write(string_view s) {
    if (fwrite(s.data(), s.size(), 1, f) != 1) {
      throw fs::filesystem_error("can't write to file"s, name, {});
    }
  }

  operator bool() const { return f != nullptr; }
  FILE* get() const { return f; }

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

  // Returns newline-ended line.
  std::string readLine() {
    char buf[512];
    buf[0] = 0;
    bool ok = fgets(buf, sizeof(buf), get());
    if (!ok) { return ""; }  // EOF or error
    string line = buf;
    if (line.empty() || line.back() != '\n') {
      log("%s : line \"%s\" does not end with a newline", name.c_str(), line.c_str());
      throw "lines must end with newline";
    }
    return line;
  }

  std::optional<std::string> maybeReadLine() {
    std::string line = readLine();
    if (line.empty()) { return std::nullopt; }
    return line;
  }

  template<typename T>
  std::vector<T> read(u32 nWords) {
    vector<T> ret;
    ret.resize(nWords);
    read(ret.data(), nWords * sizeof(T));
    return ret;
  }

  void read(void* data, u32 nBytes) {
    if (!fread(data, nBytes, 1, get())) { throw(std::ios_base::failure(name + ": can't read")); }
  }

  u32 readUpTo(void* data, u32 nUpToBytes) { return fread(data, 1, nUpToBytes, get()); }
  
  string readAll() {
    size_t sz = size();
    return {read<char>(sz).data(), sz};
  }
};

// Copyright (C) Mihai Preda.

#pragma once

#include "common.h"
#include "log.h"

#include <cstdio>
#include <cstdarg>
#include <cassert>
#include <unistd.h>
#include <filesystem>
#include <vector>
#include <string>
#include <optional>

#if defined(_WIN32) || defined(__WIN32__)
#include <io.h>
#endif

#if defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE)
#define HAS_SETLINEBUF 1
#else
#define HAS_SETLINEBUF 0
#endif

namespace fs = std::filesystem;

inline fs::path operator+(fs::path p, const std::string& tail) {
  p += tail;
  return p;
}

class File {
  FILE* f = nullptr;
  const bool readOnly;
  
  File(const fs::path &path, const string& mode, bool throwOnError);

  bool readNoThrow(void* data, u32 nBytes) { return fread(data, nBytes, 1, get()); }
  
  void read(void* data, u32 nBytes) {
    if (!readNoThrow(data, nBytes)) { throw(std::ios_base::failure(name + ": can't read")); }
  }

  void datasync() {
    fflush(f);
#if defined(_WIN32) || defined(__WIN32__)
    _commit(fileno(f));
#else
    fdatasync(fileno(f));
#endif
  }
  
public:
  const std::string name;

  static i64 size(const fs::path& name);

  static File openRead(const fs::path& name) { return File{name, "rb", false}; }
  static File openReadThrow(const fs::path& name) { return File{name, "rb", true}; }
  
  static File openWrite(const fs::path& name) { return File{name, "wb", true}; }
  
  static File openAppend(const fs::path &name) { return File{name, "ab", true}; }
  
  static void append(const fs::path& name, std::string_view text) { File::openAppend(name).write(text); }

  File() : f{}, readOnly{true} {}

  File(FILE* f, const string& name) : f{f}, readOnly{false}, name{name} {}
  
  File(File&& other) : f{other.f}, readOnly{other.readOnly}, name{other.name} { other.f = nullptr; }
  
  File& operator=(File&& other);

  File(const File& other) = delete;
  File& operator=(const File& other) = delete;

  ~File();
  
  class It {
  public:
    explicit It(File& file) : file{&file}, line{file ? file.maybeReadLine() : nullopt} {}
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
  
  template<typename T>
  void write(const vector<T>& v) const { write(v.data(), v.size() * sizeof(T)); }

  template<typename T>
  void write(const T& x) const { write(&x, sizeof(T)); }

  void write(const void* data, u32 nBytes) const {
    if (!fwrite(data, nBytes, 1, get())) { throw(std::ios_base::failure((name + ": can't write data").c_str())); }
  }
  
  void seek(long offset, int whence = SEEK_SET) {
    int ret = fseek(get(), offset, whence);
    if (ret) {
      throw(std::ios_base::failure(("fseek: "s + to_string(ret)).c_str()));
    }
  }

  void flush() { fflush(get()); }
  
  int printf(const char *fmt, ...) const __attribute__((format(printf, 2, 3))) {
    va_list va;
    va_start(va, fmt);
    int ret = vfprintf(f, fmt, va);
    va_end(va);

#if !HAS_LINEBUF
    fflush(f);
#endif

    return ret;
  }

  int scanf(const char *fmt, ...) __attribute__((format(scanf, 2, 3))) {
    va_list va;
    va_start(va, fmt);
    int ret = vfscanf(f, fmt, va);
    va_end(va);
    return ret;
  }
  
  void write(const string& s) { write(string_view(s)); }
  void write(const char* s) { write(string_view(s)); }
  void write(string_view s) { write(s.data(), s.size()); }

  operator bool() const { return f != nullptr; }
  FILE* get() const { return f; }

  long ftell() const {
    long pos = ::ftell(get());
    assert(pos >= 0);
    return pos;
  }

  long seekEnd() {
    seek(0, SEEK_END);
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
    char buf[1024];
    buf[0] = 0;
    bool ok = fgets(buf, sizeof(buf), get());
    if (!ok) { return ""; }  // EOF or error
    string line = buf;
    if (line.empty() || line.back() != '\n') {
      log("%s : line \"%s\" does not end with a newline\n", name.c_str(), line.c_str());
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

  template<typename T>
  std::vector<T> readChecked(u32 nWords) {
    u32 expectedCRC = read<u32>(1)[0];
    return readWithCRC<T>(nWords, expectedCRC);
  }

  template<typename T>
  void writeChecked(const vector<T>& data) {
    write(u32(crc32(data)));
    write(data);
  }

  template<typename T>
  std::vector<T> readWithCRC(u32 nWords, u32 crc) {
    auto data = read<T>(nWords);
    if (crc != crc32(data)) {
      log("File '%s' : CRC: expected %u, actual %u\n", name.c_str(), crc, crc32(data));
      throw "CRC";
    }    
    return data;
  }

  std::vector<u32> readBytesLE(u32 nBytes) {
    assert(nBytes > 0);
    u32 nWords = (nBytes - 1) / 4 + 1;
    vector<u32> data(nWords);
    read(data.data(), nBytes);
    return data;
  }
  
  u32 readUpTo(void* data, u32 nUpToBytes) { return fread(data, 1, nUpToBytes, get()); }
  
  string readAll() {
    size_t sz = size();
    return {read<char>(sz).data(), sz};
  }
};

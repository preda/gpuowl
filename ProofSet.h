// Copyright (C) Mihai Preda.

#pragma once

#include "File.h"
#include "common.h"
#include <vector>
#include <string>
#include <filesystem>
#include <cassert>

namespace fs = std::filesystem;

class ProofSet {
  // Exponent, step
  static constexpr const char* HEADER_v1 = "OWL SET 1 %u %u\n";
  
public:
  static bool exists(u32 E) {
    std::error_code ec;
    return fs::exists(File::fileName(E, "", "set.owl"), ec);
  }
  
  ProofSet(u32 E, u32 iniPow) : E{E},
                    file{File::openReadAppend(E, "set.owl")},
                    nWords{(E - 1) / 32 + 1} {
    assert(E & ((1 << iniPow) - 1));
    std::string headerLine = file.readLine();
    if (headerLine.empty()) {
      if (!file.empty()) {
        throw std::runtime_error(file.name + ": can't read header");
      } else {    
        file.printf(HEADER_v1, E, E / (1 << iniPow));
        file.seek(0);
        headerLine = file.readLine();
        assert(!headerLine.empty());
      }
    }

    u32 fileE{};
    if (sscanf(headerLine.c_str(), HEADER_v1, &fileE, &step_) != 2) {
      throw std::runtime_error(file.name + ": Invalid header");
    }
    assert(E == fileE);
    dataStart = file.ftell();
    u64 fileSize = file.seekEnd();
    u32 residueBytes= nWords * 4;
    assert(fileSize >= dataStart);
    u64 dataBytes = fileSize - dataStart;
    size_ = dataBytes / residueBytes;
    assert(size_ * u64(residueBytes) == dataBytes);
  }


  vector<u32> read(u32 pos) {
    if (pos >= size_) { throw std::out_of_range(std::to_string(pos)); }
    file.seek(dataStart + nWords * sizeof(u32) * pos);
    return file.read<u32>(nWords);
  }

  void append(const vector<u32>& data) {
    assert(data.size() == nWords);
    file.write(data);
    ++size_;
  }

  const u32 size() const { return size_; }
  const u32 step() const { return step_; }
  const u32 pow() const {
    for (u32 p = 7; p < 11; ++p) { if (E / (1 << p) == step_) { return p; }}
    log("%u unexpected proof step %u\n", E, step_);
    throw "invalid proof step";
  }
  
  const u32 E{};
  
private:
  u32 step_{};
  u32 size_{};
  u32 dataStart{};
  File file;
  u32 nWords;
};

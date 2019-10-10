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
  
  ProofSet(u32 E) : E{E},
                    step{E / 512},
                    file{File::openReadAppend(E, "set.owl")},
                    nWords{(E - 1) / 32 + 1} {
    assert(E & (512 - 1));
    std::string headerLine = file.readLine();
    if (headerLine.empty()) {
      if (!file.empty()) {
        throw std::runtime_error(file.name + ": can't read header");
      } else {    
        file.printf(HEADER_v1, E, step);
        file.seek(0);
        headerLine = file.readLine();
        assert(!headerLine.empty());
      }
    }

    u32 fileE{}, fileStep{};
    if (sscanf(headerLine.c_str(), HEADER_v1, &fileE, &fileStep) != 2) {
      throw std::runtime_error(file.name + ": Invalid header");
    }
    assert(E == fileE && step == fileStep);
    dataStart = file.ftell();
    u32 fileSize = file.seekEnd();
    u32 residueBytes= nWords * 4;
    assert(fileSize >= dataStart);
    u32 dataBytes = fileSize - dataStart;
    size_ = dataBytes / residueBytes;
    assert(size_ * residueBytes == dataBytes);
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
  
  const u32 E{};
  const u32 step{};
  
private:
  u32 size_{};
  u32 dataStart{};
  File file;
  u32 nWords;
};

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
  // Exponent, iteration, crc
  static constexpr const char* HEADER_v1 = "OWL SET 1 %u %u %08x\n";

  fs::path fileName(u32 k) {
    string sE = to_string(E);
    auto dir = fs::current_path() / sE / "set";
    fs::create_directories(dir);
    return dir / to_string(k);
  }

  
  bool validAt(u32 k) { return !readAtK(k).empty(); }
  
public:
  bool validUpTo(u32 maxK) {
    for (u32 k = step; k <= maxK; k += step) {
      if (!validAt(k)) { return false; }
    }
    return true;
  }
  
  ProofSet(u32 E, u32 power) : E{E}, power{power}, step{power ? E/1024*1024/(1 << power) : u32(-1)}, nWords{(E - 1) / 32 + 1} {
    assert(power <= 10 && (power == 0 || power >= 7));
  }


  vector<u32> readAtK(u32 k) {
    assert(k > 0 && k < E && k % step == 0);
    
    auto file{File::openRead(fileName(k))};
    if (!file) {
      log("'%s' can't open\n", file.name.c_str());
      return {};
    }
    
    auto header = file.readLine();
    u32 fileE = 0, fileK = 0, fileCrc = 0;
    if (sscanf(header.c_str(), HEADER_v1, &fileE, &fileK, &fileCrc) != 3 || fileE != E || fileK != k) {
      log("'%s' invalid header\n", file.name.c_str());
      return {};
    }
    
    std::vector<u32> data;
    try {
      data = file.read<u32>(nWords);
    } catch (ios_base::failure& e) {
      log("'%s' can't read data\n", file.name.c_str());
      return {};
    }
    
    if (crc32(data) != fileCrc) {
      log("'%s' mismatched CRC\n", file.name.c_str());
      return {};
    }

    return data;
  }

  /*
  vector<u32> readAtPos(u32 pos) {
    assert(pos > 0 && pos < (1 << power));
    return readAtK(pos * step);
  }

  void writeAtPos(u32 pos, const vector<u32>& data) {
  }
  */

  void writeAtK(u32 k, const vector<u32>& data) {
    assert(k > 0 && k < E && k % step == 0);
    
    auto file{File::openWrite(fileName(k))};    
    file.printf(HEADER_v1, E, k, crc32(data));
    file.write(data);
  }
  
  const u32 E{};
  const u32 power{};
  const u32 step{};
  
private:
  const u32 nWords{};
  
  static u32 crc32(const void *data, size_t size) {
    u32 tab[16] =
      { 0x00000000, 0x1DB71064, 0x3B6E20C8, 0x26D930AC,
        0x76DC4190, 0x6B6B51F4, 0x4DB26158, 0x5005713C,
        0xEDB88320, 0xF00F9344, 0xD6D6A3E8, 0xCB61B38C,
        0x9B64C2B0, 0x86D3D2D4, 0xA00AE278, 0xBDBDF21C,
      };
    u32 crc = ~0u;
    for (auto *p = static_cast<const u8*>(data), *end = p + size; p < end; ++p) {
      crc = tab[(crc ^  *p      ) & 0xf] ^ (crc >> 4);
      crc = tab[(crc ^ (*p >> 4)) & 0xf] ^ (crc >> 4);
    }
    return ~crc;
  }

  template<typename T>
  static u32 crc32(const std::vector<T>& v) { return crc32(v.data(), v.size() * sizeof(T)); }
};

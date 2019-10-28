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
public:
  bool shouldPersist(u32 k) {
    assert((k % blockSize) == 0);
    return power && (k % step == 0);
  }
  
  ProofSet(u32 E, u32 blockSize, u32 proofPow) : E{E}, blockSize{blockSize}, power{proofPow} {
    assert(!power || (power >= 7 && power <= 9));
  }
  
  const u32 E;
  const u32 blockSize;
  const u32 power;
  const u32 step{E / (blockSize * (1 << power)) * blockSize};
  
private:
};


  /*
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
  */

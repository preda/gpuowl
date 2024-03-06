// Copyright (C) Mihai Preda

#pragma once

#include "common.h"

#include <cinttypes>

class File;

class PRPState {
  // E, k, block-size, res64, nErrors
  static constexpr const char *PRP_v10 = "OWL PRP 10 %u %u %u %016" SCNx64 " %u\n";

  // Exponent, iteration, block-size, res64, nErrors
  // B1, nBits, start, nextK, crc
  static constexpr const char *PRP_v11 = "OWL PRP 11 %u %u %u %016" SCNx64 " %u %u %u %u %u %u\n";

  // E, k, block-size, res64, nErrors, CRC
  static constexpr const char *PRP_v12 = "OWL PRP 12 %u %u %u %016" SCNx64 " %u %u\n";

public:
  u32 k{};
  u32 blockSize{};
  u64 res64{};
  vector<u32> check;
  u32 nErrors{};

  // PRPState(u32 k, u32 blockSize, u64 res64, vector<)
  PRPState(File&& f);
  void saveTo(const File& f);
};

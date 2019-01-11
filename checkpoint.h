// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"

#include <vector>
#include <string>

u64 residue(const vector<u32> &words);

class PRPState {
  // Exponent, iteration, B1, block-size, res64.
  static constexpr const char *HEADER_v7 = "OWL PRP 7 %u %u %u %u %016llx\n";

  // Exponent, iteration, B1, block-size, res64, stage, nBitsBase
  static constexpr const char *HEADER_v8 = "OWL PRP 8 %u %u %u %u %016llx %u %u\n";

  // Exponent, iteration, block-size, res64
  static constexpr const char *HEADER_v9 = "OWL PRP 9 %u %u %u %016llx\n";

  
  static constexpr const char *SUFFIX = "";
  
  void loadInt(u32 E, u32 iniBlockSize);
  bool saveImpl(u32 E, const string &name);
  string durableName();
  
public:  
  u32 k;
  u32 blockSize;
  u64 res64;

  vector<u32> check;

  static PRPState load(u32 E, u32 iniBlockSize) {
    PRPState prp;
    prp.loadInt(E, iniBlockSize);
    return prp;
  }
  
  void save(u32 E);
};

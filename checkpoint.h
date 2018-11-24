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

  static constexpr const char *SUFFIX = "";
  
  // bool loadV7(u32 E, u32 B1, u32 iniBlockSize);
  void loadInt(u32 E, u32 B1, u32 iniBlockSize);
  bool saveImpl(u32 E, const string &name);
  string durableName();
  
public:  
  u32 k;
  u32 B1;
  u32 blockSize;
  u64 res64;
  u32 stage;

  vector<bool> basePower; // Stage-0 P-1 powerSmooth(B1).
  
  vector<u32> check;
  vector<u32> base;
  vector<u32> gcdAcc;

  static PRPState load(u32 E, u32 B1, u32 iniBlockSize) {
    PRPState prp;
    prp.loadInt(E, B1, iniBlockSize);
    return prp;
  }
  
  void save(u32 E);

  PRPState initStage1(u32 iniB1, u32 iniBlockSize, const vector<u32> &iniBase);
};

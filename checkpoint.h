// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"

#include <vector>
#include <string>

u64 residue(const vector<u32> &words);

class PRPState {
  // Exponent, iteration, block-size, res64
  static constexpr const char *HEADER_v9 = "OWL PRP 9 %u %u %u %016llx\n";  
  static constexpr const char *SUFFIX = "";
  
  void saveImpl(const string &name);
  bool load(FILE *fi);
  
public:  
  PRPState(u32 E, u32 iniBlockSize);
  PRPState(u32 E, u32 k, u32 blockSize, u64 res64, vector<u32> check)
    : E{E}, k{k}, blockSize{blockSize}, res64{res64}, check{std::move(check)} {
  }

  void save();
  
  const u32 E{};
  u32 k{};
  u32 blockSize{};
  u64 res64{};
  vector<u32> check;
};

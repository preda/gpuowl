// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"

#include <vector>
#include <string>

template<typename T> void save(u32 E, T *state);

class PRPState {
  friend void ::save<PRPState>(u32, PRPState *);

  // Exponent, iteration, B1, block-size, res64
  static constexpr const char *HEADER = "OWL PRP 7 %u %u %u %u %016llx\n";
  static constexpr const char *SUFFIX = "";
  
  bool load_v5(u32 E);
  bool load_v6(u32 E);
  void loadInt(u32 E, u32 B1, u32 iniBlockSize);
  bool saveImpl(u32 E, const string &name);
  string durableName();
  
public:  
  u32 k;
  u32 B1;
  u32 blockSize;
  u64 res64;
  vector<u32> check;
  vector<u32> base;

  static bool exists(u32 E);
  
  static PRPState load(u32 E, u32 B1, u32 iniBlockSize) {
    PRPState prp;
    prp.loadInt(E, B1, iniBlockSize);
    return prp;
  }
  
  void save(u32 E) { ::save(E, this); }
};

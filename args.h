// GpuOwL, a Mersenne primality tester. Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"
#include "clwrap.h"

#include <string>
#include <vector>

class Args {
private:
  
public:
  enum {CARRY_AUTO = 0, CARRY_SHORT, CARRY_LONG};

  std::string clArgs;
  std::string user;
  std::string cpu;
  std::string dump;
  
  std::vector<u32> devices;
  bool timeKernels = false;
  int carry = CARRY_AUTO;
  u32 blockSize = 400;
  int fftSize = 0;
  bool enableTF = false;
  u32 D = 0; // P-1 D (number of blocks)
  
  // return false to stop.
  bool parse(int argc, char **argv);
};

// Copyright Mihai Preda.

#pragma once

#include "common.h"

#include <string>
#include <vector>

class Args {
public:
  static std::string mergeArgs(int argc, char **argv);

  enum {CARRY_AUTO = 0, CARRY_SHORT, CARRY_LONG};

  void parse(string line);
  
  string user;
  string cpu;
  string dump;
  string dir;
  string resultsFile = "results.txt";
  std::vector<std::string> flags;
  
  int device = -1;
  
  bool timeKernels = false;
  bool enableTF = false;
  bool cudaYield = false;
  
  int carry = CARRY_AUTO;
  u32 blockSize = 500;
  u32 logStep = 50000;
  int fftSize = 0;

  u32 B1 = 500000;
  u32 B2 = 0;
  u32 B2_B1_ratio = 30;

  u32 prpExp = 0;
  u32 pm1Exp = 0;

  u32 maxBuffers = 0;
  size_t maxAlloc = 0;

  u32 iters = 0;

  void printHelp();
};

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
  string masterDir;
  std::vector<std::string> flags;
  
  int device = 0;
  
  bool timeKernels = false;
  bool enableTF = false;
  bool cudaYield = false;
  bool cleanup = false;
  u32 proofPow = 0;
  
  int carry = CARRY_AUTO;
  const u32 blockSize = 400;
  u32 logStep = 200000;
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

// Copyright Mihai Preda.

#pragma once

#include "common.h"

#include <string>
#include <set>

class Args {
public:
  static std::string mergeArgs(int argc, char **argv);

  enum {CARRY_AUTO = 0, CARRY_SHORT, CARRY_LONG};

  void parse(string line);
  void setDefaults();
  bool uses(const std::string& key) const { return flags.count(key); }
  
  string user;
  string cpu;
  string dump;
  string dir;
  string resultsFile = "results.txt";
  
  string masterDir;
  
  string uid;
  string binaryFile;
  string verifyPath;
  std::set<std::string> flags;
  
  int device = 0;
  
  bool timeKernels = false;
  bool cudaYield = false;
  bool cleanup = false;
  bool noSpin = false;
  bool safeMath = false;

  // Proof-related
  u32 proofPow = 8;
  string tmpDir = "";
  string proofResultDir = "proof";
  bool keepProof = false;

  int carry = CARRY_AUTO;
  u32 blockSize = 0;
  u32 logStep   = 0;
  u32 jacobiStep = 0;
  string fftSpec;

  u32 B1 = 1000000;
  u32 B2 = 0;
  u32 B2_B1_ratio = 30;

  u32 prpExp = 0;
  u32 pm1Exp = 0;
  u32 llExp = 0;
  
  u32 maxBuffers = 0;
  size_t maxAlloc = 0;

  u32 iters = 0;

  void printHelp();
};

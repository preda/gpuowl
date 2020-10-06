// Copyright Mihai Preda.

#pragma once

#include "common.h"

#include <string>
#include <set>
#include <filesystem>

namespace fs = std::filesystem;

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

  u32 proofPow = 8;

  fs::path resultsFile = "results.txt";
  fs::path masterDir;
  fs::path tmpDir = ".";
  fs::path proofResultDir = "proof";
  
  bool keepProof = false;

  int carry = CARRY_AUTO;
  u32 blockSize = 0;
  u32 logStep   = 0;
  u32 jacobiStep = 0;
  string fftSpec;

  u32 B1 = 0;
  u32 B2 = 0;
  u32 B2_B1_ratio = 20;

  u32 prpExp = 0;
  u32 pm1Exp = 0;
  
  size_t maxAlloc = 0;

  u32 iters = 0;
  u32 nSavefiles = 10;

  void printHelp();
};

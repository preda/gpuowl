// Copyright Mihai Preda.

#pragma once

#include "common.h"

#include <string>
#include <map>
#include <filesystem>

namespace fs = std::filesystem;

class Args {
public:
  static std::string mergeArgs(int argc, char **argv);

  enum {CARRY_AUTO = 0, CARRY_SHORT, CARRY_LONG};

  void parse(const string& line);
  void setDefaults();
  bool uses(const std::string& key) const { return flags.find(key) != flags.end(); }
  int value(const std::string& key) const;
  
  string user;
  string cpu;
  string dump;
  string dir;
  
  string uid;
  string binaryFile;
  string verifyPath;
  std::map<std::string, std::string> flags;
  
  int device = 0;
  
  bool timeKernels = false;
  bool cudaYield = false;
  bool noSpin = false;
  bool safeMath = true;
  bool clean = true;
  
  u32 proofPow = 8;
  u32 proofVerify = 9;

  fs::path resultsFile = "results.txt";
  fs::path masterDir;
  fs::path tmpDir = ".";
  fs::path proofResultDir = "proof";
  fs::path proofToVerifyDir = "proof-tmp";
  fs::path mprimeDir = ".";

  bool keepProof = false;

  int carry = CARRY_AUTO;
  u32 blockSize = 400;
  u32 logStep   = 0;
  string fftSpec;

  u32 B1 = 2'000'000;
  u32 B2 = 0;
  u32 B2_B1_ratio = 20;
  u32 D = 0;
  
  u32 prpExp = 0;
  
  size_t maxAlloc = 0;

  u32 iters = 0;
  u32 nSavefiles = 20;
  u32 startFrom = u32(-1);
  
  void printHelp();
};

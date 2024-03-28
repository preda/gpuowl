// Copyright Mihai Preda.

#pragma once

#include "common.h"

#include <string>
#include <map>
#include <filesystem>

namespace fs = std::filesystem;

class Args {
private:
    int proofPow = -1;

public:
  static std::string mergeArgs(int argc, char **argv);

  enum {CARRY_AUTO = 0, CARRY_SHORT, CARRY_LONG};

  Args(bool silent = false) : silent{silent} {}
  
  void parse(const string& line);
  void setDefaults();
  bool uses(const std::string& key) const { return flags.find(key) != flags.end(); }
  int value(const std::string& key, int valNotFound = -1) const;
  void readConfig(const fs::path& path);
  u32 getProofPow(u32 exponent) const;
  
  bool silent;
  string user;
  string cpu;
  string dump;
  string dir;
  
  string uid;
  string verifyPath;
  std::map<std::string, std::string> flags;
  
  int device = 0;
  
  bool safeMath = true;
  bool clean = true;
  bool verbose = false;
  bool useCache = false;

  u32 proofVerify = 10;

  fs::path resultsFile = "results.txt";
  fs::path masterDir;
  fs::path proofResultDir = "proof";
  fs::path proofToVerifyDir = "proof-tmp";
  fs::path cacheDir = "kernel-cache";

  bool keepProof = false;

  int carry = CARRY_AUTO;
  u32 workers = 1;
  u32 blockSize = 400;
  u32 logStep   = 0;
  string fftSpec;

  u32 prpExp = 0;
  u32 llExp = 0;
  
  size_t maxAlloc = 0;

  u32 iters = 0;
  u32 nSavefiles = 20;
  
  void printHelp();
};

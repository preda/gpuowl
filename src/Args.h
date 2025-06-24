// Copyright Mihai Preda.

#pragma once

#include "common.h"

#include <string>
#include <map>
#include <filesystem>

namespace fs = std::filesystem;

using KeyVal = std::pair<std::string, std::string>;

class Args {
private:
  int proofPow = -1;

public:
  static vector<KeyVal> splitArgLine(const std::string& inputLine);
  static vector<KeyVal> splitUses(std::string ss);
  static std::string mergeArgs(int argc, char **argv);

  enum {CARRY_AUTO = 0, CARRY_SHORT, CARRY_LONG};

  explicit Args(bool silent = false) : silent{silent} {}
  
  void parse(const string& line);
  void setDefaults();
  bool uses(const std::string& key) const { return flags.find(key) != flags.end(); }
  int value(const std::string& key, int valNotFound = -1) const;
  void readConfig(const fs::path& path);
  u32 getProofPow(u32 exponent) const;
  string tailDir() const;

  bool hasFlag(const string& key) const;

  bool silent;
  string user;
  string dump;
  string dir;
  
  string uid;
  string verifyPath;

  vector<string> ctune;

  bool doCtune{};
  bool doTune{};
  bool doZtune{};
  bool carryTune{};
  bool logROE{};

  std::map<std::string, std::string> flags;
  std::map<std::string, vector<KeyVal>> perFftConfig;
  
  int device = 0;
  
  bool safeMath = true;
  bool clean = true;
  bool verbose = false;
  bool useCache = false;
  bool profile = false;

  fs::path sharedConfigDir;
  fs::path proofResultDir = "proof";
  fs::path proofToVerifyDir = "proof-tmp";
  fs::path cacheDir = "kernel-cache";
  // fs::path resultsFile = "results.txt";
  // fs::path tuneFile = "tune.txt";

  bool keepProof = false;

  int carry = CARRY_AUTO;
  u32 workers = 1;
  u32 blockSize = 1000;
  u32 logStep = 20000;
  string fftSpec;

  u32 prpExp = 0;
  u32 llExp = 0;
  
  size_t maxAlloc = 0;

  u32 iters = 0;
  u32 nSavefiles = 4;

  // Extend the range of the FFTs beyond what's safe WRT ROE and CARRY32.
  // The FFT will handle up to fft.maxExp() * fftOverdrive
  // May also take values <1 to lower the max E handled.
  double fftOverdrive = 1;
  
  void printHelp();
};

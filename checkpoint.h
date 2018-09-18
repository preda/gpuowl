// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"

#include <vector>
#include <string>

string fileName(int E, const string &suffix = "");

template<typename T> void save(u32 E, T *state) {
  string tempFile = fileName(E, "-temp"s + T::SUFFIX);
  if (!state->saveImpl(E, tempFile)) {
    throw "can't save";
  }
  
  string prevFile = fileName(E, "-prev"s + T::SUFFIX);
  remove(prevFile.c_str());
  
  string saveFile = fileName(E, T::SUFFIX);
  rename(saveFile.c_str(), prevFile.c_str());
  rename(tempFile.c_str(), saveFile.c_str());
  
  string persist = state->durableName();
  if (!persist.empty() && !state->saveImpl(E, fileName(E, persist + T::SUFFIX))) {
    throw "can't save";
  }
}

class PRPFState {
  friend void ::save<PRPFState>(u32, PRPFState *);

  // E, k, B1, blockSize, res64
  static constexpr const char *HEADER = "OWL PRPF 1 %u %u %u %u %016llx\n";
  static constexpr const char *SUFFIX = ".prpf";
  
  void loadInt(u32 E, u32 iniB1, u32 iniBlockSize);
  bool saveImpl(u32 E, string name);
  string durableName() { return (k % 20'000'000 == 0) ? "."s + to_string(k/1000000)+"M":""s; }
  
public:
  u32 k;
  u32 B1;
  u32 blockSize;
  u64 res64;
  vector<u32> base;
  vector<u32> check;

  // P-1 must be completed before PRP-1 may start.
  static bool canProceed(u32 E, u32 B1);

  static PRPFState load(u32 E, u32 B1, u32 blockSize) {
    PRPFState prpf;
    prpf.loadInt(E, B1, blockSize);
    return prpf;
  }
  
  void save(u32 E) { ::save(E, this); }
};

class PRPState {
  friend void ::save<PRPState>(u32, PRPState *);

  // Exponent, iteration, block-size, res64, nErrors.
  static constexpr const char *HEADER = "OWL PRP 6 %u %u %u %016llx %u\n";
  static constexpr const char *SUFFIX = ".prp";
  
  bool load_v5(u32 E);
  void loadInt(u32 E, u32 iniBlockSize);
  bool saveImpl(u32 E, const string &name);
  string durableName();
  
public:  
  u32 k;
  u32 blockSize;
  u32 nErrors;
  u64 res64;
  vector<u32> check;

  static PRPState load(u32 E, u32 iniBlockSize) {
    PRPState prp;
    prp.loadInt(E, iniBlockSize);
    return prp;
  }
  
  void save(u32 E) { ::save(E, this); }
};

class PFState {
  friend void ::save<PFState>(u32, PFState *);
  
  // Exponent, iteration, total-iterations, B1.
  static constexpr const char *HEADER = "OWL PF 1 %u %u %u %u\n";
  static constexpr const char *SUFFIX = ".pf";

  void loadInt(u32 E, u32 iniB1);
  bool saveImpl(u32 E, const string &name);
  string durableName() { return ""; }

public:
  u32 k;
  u32 kEnd;
  u32 B1;
  vector<u32> base;

  static PFState load(u32 E, u32 B1) {
    PFState pf;
    pf.loadInt(E, B1);
    return pf;
  }

  void save(u32 E) { ::save(E, this); }
  bool isCompleted() { return k > 0 && k == kEnd; }
};

class TFState {
  friend void ::save<TFState>(u32, TFState *);

  // Exponent, bitLo, bitHi, classDone, classTotal.
  static constexpr const char *HEADER = "OWL TF 2 %u %u %u %u %u\n";
  static constexpr const char *SUFFIX = ".tf";
  
  void loadInt(u32 E);
  string durableName() { return ""; }

  bool saveImpl(u32 E, string name);
  
public:
  u32 bitLo, bitHi;
  u32 nDone, nTotal;

  static TFState load(u32 E) {
    TFState tf;
    tf.loadInt(E);
    return tf;
  }
  
  void save(u32 E) { ::save(E, this); }
};

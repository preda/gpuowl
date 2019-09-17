// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"

#include <vector>
#include <string>

u64 residue(const vector<u32> &words);

class StateLoader {
protected:
  virtual ~StateLoader() = default;
  
  virtual bool doLoad(const char* headerLine, FILE* fi) = 0;
  virtual void doSave(FILE* fo) = 0;

  bool load(u32 E, const std::string& extension);
  void save(u32 E, const std::string& extension);
  
  bool load(FILE* fi) {
    char line[256];
    if (!fgets(line, sizeof(line), fi)) { return false; }
    return doLoad(line, fi);
  }
};

class PRPState : private StateLoader {
  // Exponent, iteration, block-size, res64
  static constexpr const char *HEADER_v9  = "OWL PRP 9 %u %u %u %016llx\n";
  
  // Exponent, iteration, block-size, res64, nErrors
  static constexpr const char *HEADER_v10 = "OWL PRP 10 %u %u %u %016llx %u\n";
  
protected:
  bool doLoad(const char* headerLine, FILE *fi) override;
  void doSave(FILE* fo) override;
  
public:  
  PRPState(u32 E, u32 iniBlockSize);
  PRPState(u32 E, u32 k, u32 blockSize, u64 res64, vector<u32> check, u32 nErrors)
    : E{E}, k{k}, blockSize{blockSize}, res64{res64}, check{std::move(check)}, nErrors{nErrors} {
  }

  void save() { StateLoader::save(E, "owl"); }
  
  const u32 E{};
  u32 k{};
  u32 blockSize{};
  u64 res64{};
  vector<u32> check;
  u32 nErrors{};
};

class Pm1State : private StateLoader {
  // Exponent, B1, iteration, nBits
  static constexpr const char *HEADER_v1 = "OWL PM1 1 %u %u %u %u\n";
  
  bool doLoad(const char* headerLine, FILE *fi) override;
  void doSave(FILE *fo) override;
  
public:
  Pm1State(u32 E);
  Pm1State(u32 E, u32 B1, u32 k, u32 nBits, vector<u32> data)
    : E{E}, B1{B1}, k{k}, nBits{nBits}, data{std::move(data)} {
  }

  void save() { StateLoader::save(E, "pm1"); }

  const u32 E;
  u32 B1;
  u32 k;
  u32 nBits;
  vector<u32> data;
};

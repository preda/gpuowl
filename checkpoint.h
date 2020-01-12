// GpuOwl Mersenne primality tester; Copyright (C) Mihai Preda.

#pragma once

#include "File.h"
#include "common.h"

#include <vector>
#include <string>
#include <cinttypes>

u64 residue(const vector<u32> &words);

void deleteSaveFiles(u32 E);

class StateLoader {
protected:
  virtual ~StateLoader() = default;
  
  virtual bool doLoad(const char* headerLine, FILE* fi) = 0;
  virtual void doSave(FILE* fo) = 0;
  virtual u32 getK() = 0;
  
  bool load(u32 E, const std::string& extension);
  void save(u32 E, const std::string& extension, u32 k = 0);
  
  bool load(FILE* fi) {
    char line[256];
    if (!fgets(line, sizeof(line), fi)) { return false; }
    return doLoad(line, fi);
  }
};

class PRPState : private StateLoader {
  // Exponent, iteration, block-size, res64, nErrors
  static constexpr const char *HEADER_v10 = "OWL PRP 10 %u %u %u %016" SCNx64 " %u\n";
  static constexpr const char *EXT = "owl";
  
protected:
  bool doLoad(const char* headerLine, FILE *fi) override;
  void doSave(FILE* fo) override;
  u32 getK() override { return k; }
  
public:  
  static void cleanup(u32 E);

  PRPState(u32 E, u32 iniBlockSize);
  PRPState(u32 E, u32 k, u32 blockSize, u64 res64, vector<u32> check, u32 nErrors)
    : E{E}, k{k}, blockSize{blockSize}, res64{res64}, check{std::move(check)}, nErrors{nErrors} {
  }

  void save(bool persist) { StateLoader::save(E, EXT, persist ? k : 0); }

  const u32 E{};
  u32 k{};
  u32 blockSize{};
  u64 res64{};
  vector<u32> check;
  u32 nErrors{};
};

class P1State : private StateLoader {
  // Exponent, B1, iteration, nBits
  static constexpr const char *HEADER_v1 = "OWL P1 1 %u %u %u %u\n";
  static constexpr const char *EXT = "p1.owl";
  
  bool doLoad(const char* headerLine, FILE *fi) override;
  void doSave(FILE *fo) override;
  u32 getK() override { return k; }
  
public:
  static void cleanup(u32 E);

  P1State(u32 E, u32 B1);
  P1State(u32 E, u32 B1, u32 k, u32 nBits, vector<u32> data)
    : E{E}, B1{B1}, k{k}, nBits{nBits}, data{std::move(data)} {
  }

  void save() { StateLoader::save(E, EXT); }

  const u32 E;
  u32 B1;
  u32 k;
  u32 nBits;
  vector<u32> data;
};

class P2State : private StateLoader {
  // Exponent, B1, B2, nWords, kDone
  static constexpr const char *HEADER_v1 = "OWL P2 1 %u %u %u %u 2880 %u\n";
  static constexpr const char *EXT = "p2.owl";
  
  bool doLoad(const char* headerLine, FILE *fi) override;
  void doSave(FILE *fo) override;
  u32 getK() override { return k; }
  
public:
  static void cleanup(u32 E);
  
  P2State(u32 E, u32 B1, u32 B2);
  P2State(u32 E, u32 B1, u32 B2, u32 k, vector<double> raw)
    : E{E}, B1{B1}, B2{B2}, k{k}, raw{std::move(raw)} {
  }

  void save() { StateLoader::save(E, EXT); }

  const u32 E;
  u32 B1;
  u32 B2;
  u32 k;
  vector<double> raw;
};

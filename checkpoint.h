// GpuOwl Mersenne primality tester; Copyright (C) Mihai Preda.

#pragma once

#include "File.h"
#include "common.h"

#include <vector>
#include <string>
#include <cinttypes>

void deleteSaveFiles(u32 E);

class StateLoader {
protected:
  virtual ~StateLoader() = default;

  bool load(u32 E, const std::string& extension);
  void save(u32 E, const std::string& extension, u32 k = 0);
  
  virtual bool doLoad(const char* headerLine, FILE* fi) = 0;
  virtual void doSave(FILE* fo) = 0;
  virtual u32 getK() = 0;
    
  virtual bool loadFile(FILE* fi) {
    char line[256];
    if (!fgets(line, sizeof(line), fi)) { return false; }
    return doLoad(line, fi);
  }
};

class B1State {
public:
  B1State() {}
  
  B1State(const B1State&)=delete;
  B1State& operator=(const B1State&)=delete;
  
  B1State(B1State&&)=default;
  B1State& operator=(B1State&&)=default;

  bool isCompleted() const { return nextK == 0; }
  bool isEmpty() const { return b1 == 0; }
  // bool isA() { return nextK > 1; }
  
  u32 b1{};
  u32 nBits{};
  u32 nextK{};
  vector<u32> data;

  bool load(u32 E, FILE *fi);
  void save(FILE *fo);
};

class PRPState : private StateLoader {
  // Exponent, iteration, block-size, res64, nErrors
  static constexpr const char *HEADER_v10 = "OWL PRP 10 %u %u %u %016" SCNx64 " %u\n";

  // Now PRP is running at the same time with up to two P-1 first stage.
  // Exponent, iteration, block-size, res64, nErrors
  static constexpr const char *HEADER_v11 = "OWL PRP 11 %u %u %u %016" SCNx64 " %u\n";
  // %u %u %u %u %u %u\n";
  
  static constexpr const char *EXT = "owl";
  
protected:
  bool doLoad(const char* headerLine, FILE *fi) override;
  void doSave(FILE* fo) override;
  u32 getK() override { return k; }
  
public:  
  static void cleanup(u32 E);

  PRPState(u32 E, u32 iniBlockSize);
  
  PRPState(u32 E, u32 k, u32 blockSize, u64 res64, const vector<u32>& check, u32 nErrors)
    : E{E}, k{k}, blockSize{blockSize}, res64{res64}, check{check}, nErrors{nErrors} {
  }

  void save(bool persist) { StateLoader::save(E, EXT, persist ? k : 0); }

  const u32 E{};
  u32 k{};
  u32 blockSize{};
  u64 res64{};
  vector<u32> check;
  u32 nErrors{};

  B1State highB1{};
  B1State lowB1{};
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

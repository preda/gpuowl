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

struct PRPState {
  // E, k, block-size, res64, nErrors
  static constexpr const char *HEADER_v10 = "OWL PRP 10 %u %u %u %016" SCNx64 " %u\n";

  // Exponent, iteration, block-size, res64, nErrors
  // B1, nBits, start, nextK, crc
  static constexpr const char *HEADER_v11 = "OWL PRP 11 %u %u %u %016" SCNx64 " %u %u %u %u %u %u\n";

  // E, k, block-size, res64, nErrors, crc
  static constexpr const char *HEADER_v12 = "OWL PRP 12 %u %u %u %016" SCNx64 " %u %u\n";
  
  static constexpr const char *EXT = ".owl";

  static void cleanup(u32 E);

  static PRPState load(u32 E, u32 iniBlockSize);
  static void save(u32 E, const PRPState& state);
  
  u32 k{};
  u32 blockSize{};
  u64 res64{};
  vector<u32> check;
  u32 nErrors{};

private:
  static PRPState loadInt(u32 E, u32 k);
};

struct P1State {
  // E, B1, k, nextK, crc32
  static constexpr const char *HEADER_v2 = "OWL P1 2 %u %u %u %u %u\n";
  static constexpr const char *EXT = ".p1.owl";

  u32 nextK;
  vector<u32> data;
  
  static void cleanup(u32 E);
  static P1State load(u32 E, u32 b1, u32 k);
  static void save(u32 E, u32 b1, u32 k, const P1State& p1State);
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

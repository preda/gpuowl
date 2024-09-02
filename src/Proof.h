// Copyright (C) Mihai Preda.

#pragma once

#include "File.h"
#include "common.h"

namespace fs = std::filesystem;

class Gpu;

struct ProofInfo {
  u32 power;
  u32 exp;
  string md5;
};

namespace proof {

array<u64, 4> hashWords(u32 E, const Words& words);

array<u64, 4> hashWords(u32 E, array<u64, 4> prefix, const Words& words);

string fileHash(const fs::path& filePath);

ProofInfo getInfo(const fs::path& proofFile);

}

class Proof {  
public:
  const u32 E;
  const Words B;
  const vector<Words> middles;

  // Hashes is only used during the optional self-verification
  const vector<u64> hashes;

  /*Example header:
    PRP PROOF\n
    VERSION=2\n
    HASHSIZE=64\n
    POWER=8\n
    NUMBER=M216091\n
  */
  static const constexpr char* HEADER_v2 = "PRP PROOF\nVERSION=2\nHASHSIZE=64\nPOWER=%u\nNUMBER=M%u%c";

  static Proof load(const fs::path& path);

  void save(const fs::path& proofResultDir) const;

  fs::path file(const fs::path& proofDir) const;
  
  bool verify(Gpu *gpu) const;
};

class ProofSet {
public:
  const u32 E;
  const u32 power;
  
private:  
  vector<u32> points;  
  
  bool isValidTo(u32 limitK) const;

  static bool canDo(u32 E, u32 power, u32 currentK);

  mutable decltype(points)::const_iterator cacheIt{};

  bool fileExists(u32 k) const;

  static fs::path proofPath(u32 E) { return fs::path(to_string(E)) / "proof"; }
public:
  
  static u32 bestPower(u32 E);
  static u32 effectivePower(u32 E, u32 power, u32 currentK);
  static double diskUsageGB(u32 E, u32 power);
  static bool isInPoints(u32 E, u32 power, u32 k);
  
  ProofSet(u32 E, u32 power);
    
  u32 next(u32 k) const;

  static void save(u32 E, u32 power, u32 k, const Words& words);
  static Words load(u32 E, u32 power, u32 k);
        
  void save(u32 k, const Words& words) const { return save(E, power, k, words); }
  Words load(u32 k) const { return load(E, power, k); }


  Proof computeProof(Gpu *gpu) const;
};

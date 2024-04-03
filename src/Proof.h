// Copyright (C) Mihai Preda.

#pragma once

#include "File.h"
#include "ProofCache.h"
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
  u32 E;
  Words B;
  vector<Words> middles;

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
  fs::path proofPath;
  ProofCache cache{E, proofPath};

  vector<u32> points;  
  
  bool isValidTo(u32 limitK) const;

  static bool canDo(u32 E, u32 power, u32 currentK);

  mutable decltype(points)::const_iterator cacheIt{};

public:
  
  static u32 bestPower(u32 E);
  static u32 effectivePower(u32 E, u32 power, u32 currentK);
  static double diskUsageGB(u32 E, u32 power);
  
  ProofSet(u32 E, u32 power);
    
  u32 next(u32 k) const;

  void save(u32 k, const Words& words);

  Words load(u32 k) const;
        
  Proof computeProof(Gpu *gpu) const;
};

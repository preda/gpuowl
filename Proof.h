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
    VERSION=1\n
    HASHSIZE=64\n
    POWER=8\n
    NUMBER=M216091\n
  */
  static const constexpr char* HEADER = "PRP PROOF\nVERSION=1\nHASHSIZE=64\nPOWER=%u\nNUMBER=M%u%c";

  static Proof load(const fs::path& path);
  
  fs::path save(const fs::path& proofResultDir);

  bool verify(Gpu *gpu);
};

class ProofSet {
  fs::path exponentDir;
  fs::path proofPath{exponentDir / "proof"};
  ProofCache cache{proofPath};
  
public:
  u32 E;
  u32 power;
  u32 topK{roundUp(E, (1 << power))};
  u32 step{topK / (1 << power)};

  static bool canDo(const fs::path& tmpDir, u32 E, u32 power, u32 currentK) {
    assert(power >= 6 && power <= 10);
    return ProofSet{tmpDir, E, power}.isValidTo(currentK);
  }
  
  static u32 effectivePower(const fs::path& tmpDir, u32 E, u32 power, u32 currentK) {
    for (u32 p : {power, 9u, 8u, 7u, 6u}) {
      log("validating proof residues for power %u\n", p);
      if (canDo(tmpDir, E, p, currentK)) { return p; }
    }
    return 0;
  }
  
  ProofSet(const fs::path& tmpDir, u32 E, u32 power) : exponentDir(tmpDir / to_string(E)), E{E}, power{power} {
    assert(E & 1); // E is supposed to be prime
    assert(topK % step == 0);
    assert(topK / step == (1u << power));
    
    fs::create_directory(exponentDir);
    fs::create_directory(proofPath);
  }

  void cleanup() {
    error_code noThrow;
    cache.clear();
    fs::remove_all(proofPath, noThrow);
    fs::remove(exponentDir, noThrow);
  }
  
  u32 kProofEnd(u32 kEnd) const {
    if (!power) { return kEnd; }
    assert(topK > kEnd);
    return topK;
  }
  
  u32 firstPersistAt(u32 k) const { return power ? roundUp(k, step): -1; }

  void save(u32 k, const Words& words) {
    assert(k > 0 && k <= topK);
    assert(k % step == 0);
    cache.save(k, words);
  }

  Words load(u32 k) const {
    assert(k > 0 && k <= topK);
    assert(k % step == 0);
    return cache.load(E, k);
  }
        
  bool isValidTo(u32 limitK) const {
    if (!power) { return true; }    
    try {
      for (u32 k = step; k <= limitK; k += step) { load(k); }
      return true;
    } catch (fs::filesystem_error&) {
    } catch (std::ios_base::failure&) {
    }
    return false;
  }

  // A quick-and-dirty version that checks only the first two residues.
  bool seemsValidTo(u32 limitK) const { return isValidTo(std::min(limitK, 2 * step)); }
  
  bool isComplete() const { return isValidTo(topK); }
  
  Proof computeProof(Gpu *gpu);
};

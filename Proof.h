// Copyright (C) Mihai Preda.

#pragma once

#include "File.h"
#include "common.h"

#include <unordered_map>

namespace fs = std::filesystem;

class Gpu;

namespace ProofUtil {

array<u64, 4> hashWords(u32 E, const Words& words);

array<u64, 4> hashWords(u32 E, array<u64, 4> prefix, const Words& words);

string fileHash(const fs::path& filePath);

}

struct ProofInfo {
  u32 power;
  u32 exp;
  string md5;
};

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
  
  static ProofInfo getInfo(const fs::path& proofFile) {
    string hash = ProofUtil::fileHash(proofFile);
    File fi = File::openRead(proofFile, true);
    u32 E = 0, power = 0;
    char c = 0;
    if (fi.scanf(HEADER, &power, &E, &c) != 3 || c != '\n') {
      log("Proof file '%s' has invalid header\n", proofFile.string().c_str());
      throw "Invalid proof header";
    }
    return {power, E, hash};
  }
  
  fs::path save(const fs::path& proofResultDir) {
    string strE = to_string(E);
    u32 power = middles.size();
    fs::path fileName = proofResultDir / (strE + '-' + to_string(power) + ".proof");
    File fo = File::openWrite(fileName);
    fo.printf(HEADER, power, E, '\n');
    fo.write(B.data(), (E-1)/8+1);
    for (const Words& w : middles) { fo.write(w.data(), (E-1)/8+1); }
    return fileName;
  }

  static Proof load(const fs::path& path) {
    File fi = File::openRead(path, true);
    u32 E = 0, power = 0;
    char c = 0;
    if (fi.scanf(HEADER, &power, &E, &c) != 3 || c != '\n') {
      log("Proof file '%s' has invalid header\n", path.string().c_str());
      throw "Invalid proof header";
    }
    u32 nBytes = (E - 1) / 8 + 1;
    Words B = fi.readBytesLE(nBytes);
    vector<Words> middles;
    for (u32 i = 0; i < power; ++i) { middles.push_back(fi.readBytesLE(nBytes)); }
    return {E, B, middles};
  }
  
  bool verify(Gpu *gpu);
};

class ProofCache {
  std::unordered_map<u32, Words> pending;
  fs::path proofPath;
  
  bool write(u32 k, const Words& words) {
    File f = File::openWrite(proofPath / to_string(k));
    try {
      f.write(words);
      f.write<u32>({crc32(words)});
      return true;
    } catch (fs::filesystem_error& e) {
      return false;
    }
  }

  Words read(u32 E, u32 k) const {
    File f = File::openRead(proofPath / to_string(k), true);
    vector<u32> words = f.read<u32>(E / 32 + 2);
    u32 checksum = words.back();
    words.pop_back();
    if (checksum != crc32(words)) {
      log("checksum %x (expected %x) in '%s'\n", crc32(words), checksum, f.name.c_str());
      throw fs::filesystem_error{"checksum mismatch", {}};
    }
    return words;
  }


  void flush() {
    for (auto it = pending.cbegin(), end = pending.cend(); it != end && write(it->first, it->second); it = pending.erase(it));
    if (!pending.empty()) {
      log("Could not write %u residues under '%s' -- hurry make space!\n", u32(pending.size()), proofPath.string().c_str());
    }
  }
  
public:
  ProofCache(const fs::path& proofPath) : proofPath{proofPath} {}
  
  ~ProofCache() { flush(); }
  
  void save(u32 k, const Words& words) {
    if (pending.empty() && write(k, words)) { return; }    
    pending[k] = words;
    flush();
  }

  Words load(u32 E, u32 k) const {
    auto it = pending.find(k);
    return (it == pending.end()) ? read(E, k) : it->second;
  }

  void clear() { pending.clear(); }
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

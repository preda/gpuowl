// Copyright (C) Mihai Preda.

#pragma once

#include "Gpu.h"
#include "GmpUtil.h"
#include "File.h"
#include "common.h"
#include "Sha3Hash.h"

#include <vector>
#include <string>
#include <cassert>
#include <filesystem>
#include <cinttypes>
#include <climits>

struct ProofUtil {
  static const u32 MAX_POWER = 10;
  static const u32 MULTIPLE = (1 << MAX_POWER);
  
  static Words makeWords(u32 E, u32 init) {
    u32 nWords = (E - 1) / 32 + 1;
    Words x(nWords);
    x[0] = init;
    return x;
  }

  static u32 revbin(int nBits, u32 k) {
    u32 r = 0;    
    for (u32 bit = 1 << (nBits - 1); bit; bit>>=1, k>>=1) { if (k & 1) { r |= bit; }}
    return r;
  }
};

class Proof {
public:
  u32 E;
  Words B;
  vector<Words> middles;
  u64 finalHash; // a data check

  // version(1), E, power, finalHash
  static const constexpr char* HEADER = "PRProof 1 %u %u 0x%" SCNx64 "\n";
  
  fs::path save() {
    string strE = to_string(E);
    fs::path fileName = fs::current_path() / strE / (strE + '-' + hex(finalHash).substr(0, 4) + ".proof");
    File fo = File::openWrite(fileName);
    fo.printf(HEADER, E, u32(middles.size()), finalHash);
    fo.write(B);
    for (const Words& w : middles) { fo.write(w); }
    return fileName;
  }

  static Proof load(fs::path path) {
    File fi = File::openRead(path, true);
    string headerLine = fi.readLine();
    u32 E = 0, power = 0;
    u64 finalHash = 0;
    if (sscanf(headerLine.c_str(), HEADER, &E, &power, &finalHash) != 3) {
      log("Proof file '%s' has invalid header '%s'\n", path.string().c_str(), headerLine.c_str());
      throw "Invalid proof header";
    }
    u32 nWords = (E - 1) / 32 + 1;
    Words B = fi.read<u32>(nWords);
    vector<Words> middles;
    for (u32 i = 0; i < power; ++i) { middles.push_back(fi.read<u32>(nWords)); }
    return {E, B, middles, finalHash};
  }
  
  bool verify(Gpu *gpu) {
    u32 power = middles.size();
    assert(power > 0);

    u32 topK = roundUp(E, ProofUtil::MULTIPLE);
    assert(topK % (1 << power) == 0);
    assert(topK > E);
    u32 step = topK / (1 << power);

    bool isPrime = false;
    {
      Words A{ProofUtil::makeWords(E, 3)};
      log("proof: doing %d iterations\n", topK - E + 1);
      A = gpu->expExp2(A, topK - E + 1);
      isPrime = (A == B);
      log("the proof indicates %s (%016" PRIx64 " vs. %016" PRIx64 " for a PRP)\n",
          isPrime ? "probable prime" : "composite", res64(B), res64(A));
    }

    Words A{ProofUtil::makeWords(E, 3)};
    
    auto hash = SHA3::hash(B);
    
    for (u32 i = 0; i < power; ++i) {
      Words& M = middles[i];
      hash = SHA3::hash(hash, M);
      A = gpu->expMul(A, hash[0], M);
      B = gpu->expMul(M, hash[0], B);
    }
    
    if (hash[0] != finalHash) {
      log("proof: hash %016" PRIx64 " expected %016" PRIx64 "\n", hash[0], finalHash);
      return false;
    }
    
    log("proof verification: doing %d iterations\n", step);
    A = gpu->expExp2(A, step - 1);
    u64 verificationSign = res64(A);
    log("proof: verification signature: %016" PRIx64 "\n", verificationSign);
    A = gpu->expExp2(A, 1);
    
    if (A != B) {
      log("proof: invalid (%016" PRIx64 " expected %016" PRIx64 "); verification %016" PRIx64 "\n",
          res64(A), res64(B), verificationSign);
      return false;
    }

    log("proof: %u proved %s; verification %016" PRIx64 "\n",
        E, isPrime ? "probable prime" : "composite", verificationSign);
    
    return true;
  }

private:
  static u64 res64(const Words& words) { return (u64(words[1]) << 32) | words[0]; }
};

class ProofSet {
public:  
  u32 E;
  u32 power;
  u32 topK{roundUp(E, ProofUtil::MULTIPLE)};
  u32 step{topK / (1 << power)};

  ProofSet(u32 E, u32 power) : E{E}, power{power} {
    assert(E & 1); // E is supposed to be prime
    assert(power <= ProofUtil::MAX_POWER);
    assert(topK % step == 0);
    assert(topK / step == (1u << power));
  }

  u32 kProofEnd(u32 kEnd) const { return power ? roundUp(kEnd, ProofUtil::MULTIPLE) : kEnd; }
  
  u32 firstPersistAt(u32 k) const { return power ? roundUp(k, step): -1; }

  void save(u32 k, const vector<u32>& words) {
    assert(k > 0 && k <= topK);
    assert(k % step == 0);

    File f = File::openWrite(proofPath / to_string(k));
    f.write(words);
    f.write<u32>({crc32(words)});
  }

  vector<u32> load(u32 k) const {
    assert(k > 0 && k <= topK);
    assert(k % step == 0);
    
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

  bool isValidTo(u32 limitK) const {
    if (!power) { return true; }

    fs::create_directory(proofPath);
    
    try {
      for (u32 k = step; k <= limitK; k += step) { load(k); }
    } catch (fs::filesystem_error&) {
      return false;
    }
    return true;
  }

  bool isComplete() const { return isValidTo(topK); }
  
  Proof computeProof(Gpu *gpu, u64 prpRes64) {
    assert(power > 0);
    
    Words B = load(topK);
    Words A = ProofUtil::makeWords(E, 3);

    vector<Words> middles;
    vector<u64> hashes;

    auto hash = SHA3::hash(B);

    vector<Buffer<i32>> bufVect = gpu->makeBufVector(power + 1);
    
    for (u32 p = 0; p < power; ++p) {
      auto bufIt = bufVect.begin();
      assert(p == hashes.size());
      log("proof: building level %d, hash %016" PRIx64 "\n", (p + 1), hash[0]);
      u32 s = topK / (1 << (p + 1));
      for (int i = 0; i < (1 << p); ++i) {
        Words w = load(s * (2 * i + 1));
        gpu->writeIn(*bufIt++, w);
        for (int k = 0; i & (1 << k); ++k) {
          --bufIt;
          u64 h = hashes[p - 1 - k];
          gpu->expMul(*(bufIt - 1), h, *bufIt);
        }
      }
      assert(bufIt == bufVect.begin() + 1);
      middles.push_back(gpu->readAndCompress(bufVect.front()));
      hash = SHA3::hash(hash, middles.back());
      hashes.push_back(hash[0]);
    }
    return Proof{E, std::move(B), std::move(middles), hash[0]};
  }

private:  
  static u32 crc32(const void *data, size_t size) {
    u32 tab[16] = {
                   0x00000000, 0x1DB71064, 0x3B6E20C8, 0x26D930AC,
                   0x76DC4190, 0x6B6B51F4, 0x4DB26158, 0x5005713C,
                   0xEDB88320, 0xF00F9344, 0xD6D6A3E8, 0xCB61B38C,
                   0x9B64C2B0, 0x86D3D2D4, 0xA00AE278, 0xBDBDF21C,
    };
    u32 crc = ~0;
    for (auto *p = (const unsigned char *) data, *end = p + size; p < end; ++p) {
      crc = tab[(crc ^  *p      ) & 0xf] ^ (crc >> 4);
      crc = tab[(crc ^ (*p >> 4)) & 0xf] ^ (crc >> 4);
    }
    return ~crc;
  }

  static u32 crc32(const std::vector<u32>& words) { return crc32(words.data(), sizeof(words[0]) * words.size()); }

  fs::path proofPath{fs::current_path() / to_string(E) / "proof"};
};

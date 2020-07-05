// Copyright (C) Mihai Preda.

#pragma once

#include "File.h"
#include "common.h"
#include "Sha3Hash.h"

#include <vector>
#include <string>
#include <cassert>
#include <filesystem>
#include <cinttypes>
#include <climits>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error Byte order must be Little Endian
#endif

struct ProofUtil {
  static Words makeWords(u32 E, u32 init) {
    u32 nWords = (E - 1) / 32 + 1;
    Words x(nWords);
    x[0] = init;
    return x;
  }

  static array<u64, 4> hashWords(u32 E, const Words& words) { return SHA3{}.update(words.data(), (E-1)/8+1).finish(); }
  static array<u64, 4> hashWords(u32 E, array<u64, 4> prefix, const Words& words) {
    return SHA3{}.update(prefix).update(words.data(), (E-1)/8+1).finish();
  }
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
    POWER=9\n
    NUMBER=M216091\n
  */
  static const constexpr char* HEADER = "PRP PROOF\nVERSION=1\nHASHSIZE=64\nPOWER=%u\nNUMBER=M%u%c";
  
  fs::path save() {
    string strE = to_string(E);
    u32 power = middles.size();
    fs::path fileName = fs::current_path() / strE / (strE + '-' + to_string(power) + ".proof");
    File fo = File::openWrite(fileName);
    fo.printf(HEADER, power, E, '\n');
    fo.write(B.data(), (E-1)/8+1);
    for (const Words& w : middles) { fo.write(w.data(), (E-1)/8+1); }
    return fileName;
  }

  static Proof load(fs::path path) {
    File fi = File::openRead(path, true);
    u32 E = 0, power = 0;
    char c = 0;
    if (fi.scanf(HEADER, &power, &E, &c) != 3 || c != '\n') {
      log("Proof file '%s' has invalid header\n", path.string().c_str());
      throw "Invalid proof header";
    }
    u32 nBytes = (E-1)/8+1;
    u32 nWords = (E-1)/32+1;
    Words B(nWords);
    fi.read(B.data(), nBytes);
    vector<Words> middles;
    for (u32 i = 0; i < power; ++i) {
      Words M(nWords);
      fi.read(M.data(), nBytes);
      middles.push_back(std::move(M));
    }
    return {E, B, middles};
  }
  
  bool verify(Gpu *gpu) {
    u32 power = middles.size();
    assert(power > 0);

    u32 topK = roundUp(E, (1 << power));
    assert(topK % (1 << power) == 0);
    assert(topK > E);
    u32 step = topK / (1 << power);

    bool isPrime = false;
    {
      Words A{ProofUtil::makeWords(E, 3)};
      log("proof: doing %d iterations\n", topK - E + 1);
      A = gpu->expExp2(A, topK - E + 1);
      isPrime = (A == B);
      // log("the proof indicates %s (%016" PRIx64 " vs. %016" PRIx64 " for a PRP)\n",
      //    isPrime ? "probable prime" : "composite", res64(B), res64(A));
    }

    Words A{ProofUtil::makeWords(E, 3)};
    
    auto hash = ProofUtil::hashWords(E, B);
    
    for (u32 i = 0; i < power; ++i) {
      Words& M = middles[i];
      hash = ProofUtil::hashWords(E, hash, M);
      u64 h = hash[0];
      A = gpu->expMul(A, h, M);
      B = gpu->expMul(M, h, B);
    }
    
    log("proof verification: doing %d iterations\n", step);
    A = gpu->expExp2(A, step);

    bool ok = (A == B);
    if (ok) {
      log("proof: %u proved %s\n", E, isPrime ? "probable prime" : "composite");
    } else {
      log("proof: invalid (%016" PRIx64 " expected %016" PRIx64 ")\n", res64(A), res64(B));
    }
    return ok;
  }

private:
  static u64 res64(const Words& words) { return (u64(words[1]) << 32) | words[0]; }
};

class ProofSet {
public:  
  u32 E;
  u32 power;
  u32 topK{roundUp(E, (1 << power))};
  u32 step{topK / (1 << power)};

  ProofSet(u32 E, u32 power) : E{E}, power{power} {
    assert(E & 1); // E is supposed to be prime
    assert(topK % step == 0);
    assert(topK / step == (1u << power));
  }

  u32 kProofEnd(u32 kEnd) const {
    if (!power) { return kEnd; }
    assert(topK > kEnd);
    return topK;
  }
  
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

    auto hash = ProofUtil::hashWords(E, B);

    vector<Buffer<i32>> bufVect = gpu->makeBufVector(power);
    
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
      hash = ProofUtil::hashWords(E, hash, middles.back());
      hashes.push_back(hash[0]);
    }
    return Proof{E, std::move(B), std::move(middles)};
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

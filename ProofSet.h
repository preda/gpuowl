// Copyright (C) Mihai Preda.

#pragma once

#include "Gpu.h"
#include "GmpUtil.h"
#include "File.h"
#include "common.h"
#include "Blake2.h"
#include <vector>
#include <string>
#include <cassert>
#include <filesystem>
#include <cinttypes>

struct ProofUtil {
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
  u32 E, topK;
  Words B;
  vector<Words> middles;
  u64 finalHash; // a data check

  bool verify(Gpu *gpu) {
    u32 power = middles.size();
    assert(power > 0);
    assert(topK % (1 << power) == 0);
    assert(topK <= E);
    Words saveB = B;
    
    Words A{ProofUtil::makeWords(E, 3)};
    
    u64 h = Blake2::hash({E, topK, B});
    
    for (u32 i = 0; i < power; ++i) {
      Words& M = middles[i];      
      h = Blake2::hash({h, M}); // for the final step any random value would do
      A = gpu->expMul(A, h, M);
      B = gpu->expMul(M, h, B);
    }
    
    if (h != finalHash) {
      log("proof: hash %016" PRIx64 " expected %016" PRIx64 "\n", h, finalHash);
      return false;
    }

    u32 step = topK / (1 << power);
    log("proof verification: doing %d iterations\n", step);
    A = gpu->expExp2(A, step);
    if (A != B) {
      log("proof: failed to validate: %016" PRIx64 " expected %016" PRIx64 "\n", res64(A), res64(B));
      return false;
    }

    log("proof: doing %d tail iterations\n", E - topK);
    A = gpu->expExp2(saveB, E - topK);
    bool isPrime = Gpu::equals9(A);
    u64 resType4 = res64(A);

    Gpu::doDiv9(E, A);
    log("proof: %u %s, res type1 %016" PRIx64 " type4 %016" PRIx64 "\n", E, isPrime ? "likely prime" : "proved composite", res64(A), resType4);    
    return true;
  }

private:
  static u64 res64(const Words& words) { return (u64(words[1]) << 32) | words[0]; }
};

class ProofSet {
public:
  static const u32 MAX_POWER = 10;
  static const u32 MULTIPLE = (1 << MAX_POWER);
  
  const u32 E;
  const u32 power;
  const u32 topK{E / MULTIPLE * MULTIPLE};
  const u32 step{topK / (1 << power)};

  ProofSet(u32 E, u32 power) : E{E}, power{power} {
    assert(power <= MAX_POWER);
    assert(topK % step == 0);
    assert(topK / step == (1u << power));
  }

  u32 firstPersistAfter(u32 k) const { return (power && k <= topK) ? k / step * step + step : -1; }

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

  Proof computeProof(Gpu *gpu) {
    assert(power > 0);
    
    Words B = load(topK);
    Words A = ProofUtil::makeWords(E, 3);

    vector<Words> middles;
    
    vector<mpz_class> hashes;
    hashes.emplace_back(1);
    middles.push_back(load(topK / 2));

    u64 h = Blake2::hash({E, topK, B});
    h = Blake2::hash({h, middles.back()});
    
    for (u32 p = 1; p < power; ++p) {
      log("proof: building level %d, hash %016" PRIx64 "\n", (p + 1), h);
      for (int i = 0; i < (1 << (p - 1)); ++i) { hashes.push_back(hashes[i] * h); }
      Words M = ProofUtil::makeWords(E, 1);
      u32 s = topK / (1 << (p + 1));
      for (int i = 0; i < (1 << p); ++i) {
        Words w = load(s * (2 * i + 1));
        u32 pos = ProofUtil::revbin(p, (1<<p) - 1 - i);
        M = gpu->expMul(w, bitsMSB(hashes[pos]), M);
      }
      middles.push_back(std::move(M));
      h = Blake2::hash({h, middles.back()});
    }
    return Proof{E, topK, std::move(B), std::move(middles), h};
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

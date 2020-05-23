// Copyright (C) Mihai Preda.

#pragma once

#include "GmpUtil.h"
#include "File.h"
#include "common.h"
#include "Blake2.h"
#include <vector>
#include <string>
#include <cassert>
#include <filesystem>

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

  u32 firstPersistAfter(u32 k) const { return power ? k / step * step + step : -1; }

  void save(u32 k, const vector<u32>& words) {
    assert(k > 0 && k <= topK);
    assert(k % step == 0);

    File f = File::openWrite(proofPath / to_string(k));
    f.write(words);
    f.write<u32>({crc32(words)});
  }

  vector<u32> load(u32 k) {
    assert(k > 0 && k <= topK);
    assert(k % step == 0);
    
    File f = File::openRead(proofPath / to_string(k), true);
    vector<u32> words = f.read<u32>(E / 32 + 1);
    u32 checksum = words.back();
    words.pop_back();
    if (checksum != crc32(words)) {
      log("checksum %x (expected %x) in '%s'", crc32(words), checksum, f.name.c_str());
      throw fs::filesystem_error{"checksum mismatch", {}};
    }
    return words;
  }

  bool isValidTo(u32 limitK) {
    if (!power) { return true; }
    
    try {
      for (u32 k = step; k <= limitK; k += step) { load(k); }
    } catch (fs::filesystem_error&) {
      return false;
    }
    return true;
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

using Words = std::vector<u32>;

Words makeWords3(u32 E) {
  u32 nWords = (E - 1) / 32 + 1;
  Words x(nWords);
  x[0] = 3;
  return x;
}

struct Level {
  Words x, y, u;
  u64 hash;
  vector<mpz_class> exponents;
};

struct Proof {
  u32 E, kEnd;
  Words yEnd;
  vector<Words> middles;

  bool checks() {
    u32 power = middles.size();
    assert(power);
    assert((kEnd & ((u32(1) << power) - 1)) == 0);

    Words x{makeWords3(E)};
    Words y = yEnd;
    [[maybe_unused]] u64 hash = Blake2::hash({E, kEnd, power});
    u32 distanceXY = kEnd;

    /*
    for (const Words& u : middles) {
      hash = Blake2::hash({hash, x, y, u});
      x = powMul(x, hash, u);
      y = powMul(u, hash, y);
      assert(distanceXY && ((distanceXY & 1) == 0));
      distanceXY /= 2;
    }
    */

    assert(distanceXY);
    // return exp2exp(x, distanceXY) == y;
    return true;
  }
};

class ProofBuilder {
public:
  ProofBuilder(u32 E, u32 kEnd, u32 power, Words y, Words u) : E{E}, kEnd{kEnd}, power{power} {
    Words x(nWords);
    x[0] = 3;
    vector<mpz_class> exponents{{1}};
    u64 hash = Blake2::hash({Blake2::hash({E, kEnd, power}), x, y, u});
    levels.push_back({std::move(x), std::move(y), std::move(u), hash, {{1}}});
  }

  void addLevel(Words x, Words y, Words u, const vector<mpz_class> exponents) {
    u64 hash = Blake2::hash({levels.back().hash, x, y, u});
    levels.push_back({std::move(x), std::move(y), std::move(u), hash, std::move(exponents)});
  }

  Proof getProof() {
    return {};
  }
  
  const u32 E;
  const u32 kEnd;
  const u32 power;
  const u32 nWords{(E-1) / 32 + 1};

private:
  vector<Level> levels{};

  /*

  void addNextLevel() {
    Level& prev = levels.back();
    Words x = powerMul(prev.x, prev.hash, prev.u);
    Words y = powerMul(prev.u, prev.hash, prev.y);
    // Words u = ;
  }
  */
  
};

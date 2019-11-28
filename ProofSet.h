// Copyright (C) Mihai Preda.

#pragma once

#include "GmpUtil.h"
#include "File.h"
#include "common.h"
#include "Blake2.h"
#include <vector>
#include <string>
#include <cassert>

class ProofSet {
public:
  bool shouldPersist(u32 k) {
    assert((k % blockSize) == 0);
    return power && (k % step == 0);
  }
  
  ProofSet(u32 E, u32 blockSize, u32 proofPow) : E{E}, blockSize{blockSize}, power{proofPow} {
    assert(!power || (power >= 7 && power <= 9));
  }
  
  const u32 E;
  const u32 blockSize;
  const u32 power;
  const u32 step{E / (blockSize * (1 << power)) * blockSize};
  
private:
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

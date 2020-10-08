// Copyright Mihai Preda

#pragma once

#include "Buffer.h"
#include "Memlock.h"
#include "common.h"

#include <cassert>

class Gpu;
class Args;
class Saver;

class B1Accumulator {
public:
  const u32 E;
  const u32 b1;
  const u32 nBits;

private:
  Gpu* gpu;
  Saver* saver;

  u32 maxBufs;  
  u32 nextK = 0;
  
  vector<bool> bits;
  vector<Buffer<i32>> bufs;
  vector<bool> engaged;
  
  std::optional<Memlock> memlock;
  
  u32 findFirstBitSet() const {
    assert(!bits.empty());
    for (u32 i = 0, end = bits.size(); i < end; ++i) { if (bits[i]) { return i; } }
    assert(false);
    return 0;
  }

  void release() {
    if (!bits.empty()) { bits.clear(); }
    // log("B1 %u: releasing %u bits\n", b1, u32(bits.size()));
    
    if (!bufs.empty()) {
      assert(memlock);
      log("B1 %u: releasing %u buffers\n", b1, u32(bufs.size()));
      bufs.clear();
      memlock.reset();
    }

    engaged.clear();
  }

  void alloc();
  
  vector<u32> fold();

  pair<u32,u32> findFirstPop(u32 start) {
    u32 sum = 0;
    for (u32 i = start; i < bits.size(); ++i) {
      if (bits[i]) {
        u32 n = i - start;
        assert(n < 64);
        u64 delta = u64(1) << n;
        if (sum + delta >= bufs.size()) {
          return {i, sum};
        } else {
          sum += delta;
        }
      }
    }
    return {0, sum};
  }

public:
  B1Accumulator(Gpu* gpu, Saver* saver, u32 E, u32 maxBufs);
  ~B1Accumulator() { release(); }

  u32 wantK() const { return nextK; }
  
  vector<u32> save(u32 k);
  
  void load(u32 k);
  
  template<typename T>
  void step(u32 kAt, Buffer<T>& data);
};

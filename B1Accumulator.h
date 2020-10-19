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
  const u32 N;
  
  u32 nextK = 0;
  
  vector<bool> bits;
  vector<Buffer<i32>> bufs;

  std::optional<Memlock> memlock;
  
  u32 findFirstBitSet() const;

  void alloc();
  void release();
  
  pair<u32,u32> findFirstPop(u32 start);
  void verifyRoundtrip(const Words& expected);
  
public:
  B1Accumulator(Gpu* gpu, Saver* saver, u32 E);
  ~B1Accumulator() { release(); }

  u32 wantK() const { return nextK; }
  
  vector<u32> save(u32 k);  
  void load(u32 k);

  vector<u32> fold();

  // The step() can be invoked with either a "small" buffer or a "big" buffer depending on what
  // PRP happens to have on hand.
  template<typename T>
  void step(u32 kAt, Buffer<T>& data);
};

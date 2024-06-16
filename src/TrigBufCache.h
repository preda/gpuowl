// Copyright Mihai Preda

#pragma once

#include "Buffer.h"

#include <mutex>

using double2 = pair<double, double>;
using TrigBuf = Buffer<double2>;
using TrigPtr = shared_ptr<TrigBuf>;

class StrongCache {
  vector<TrigPtr> ptrs;
  u32 pos{};

public:
  explicit StrongCache(u32 size) : ptrs(size) {}

  void add(TrigPtr ptr) {
    ptrs.at(pos) = ptr;
    if (++pos >= ptrs.size()) { pos = 0; }
  }
};

class TrigBufCache {  
  const Context* context;
  std::mutex mut;

  std::map<tuple<u32, u32>, TrigPtr::weak_type> small;
  std::map<tuple<u32, u32, u32>, TrigPtr::weak_type> middle;
  std::map<tuple<u32, u32, u32>, TrigPtr::weak_type> bhw;
  std::map<tuple<u32, u32, u32>, TrigPtr::weak_type> square;

  // The shared-pointers below keep the most recent set of buffers alive even without any Gpu instance
  // referencing them. This allows a single worker to delete & re-create the Gpu instance and still reuse the buffers.
  StrongCache smallCache{6};
  StrongCache middleCache{6};
  StrongCache bhwCache{4};
  StrongCache squareCache{4};

public:
  TrigBufCache(const Context* context) :
    context{context}
  {}

  ~TrigBufCache();

  TrigPtr smallTrig(u32 W, u32 nW);
  TrigPtr middleTrig(u32 SMALL_H, u32 MIDDLE, u32 W);
  TrigPtr trigBHW(u32 W, u32 hN, u32 BIG_H);
  TrigPtr trigSquare(u32 hN, u32 nH, u32 SMALL_H);
};

// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "file.h"
#include "common.h"

#include <cassert>
#include <vector>

class Kset {
  vector<u64> pVect;
  vector<pair<u32, u32>> kVect;

public:
  Kset(const string &name) {
    auto fi = openRead(name, true);
    u32 k;
    int n;
    while (fscanf(fi.get(), "%u %d", &k, &n) == 2) {
      for (int i = 0; i < n; ++i) {
        u64 p = 0;
        bool ok = (fscanf(fi.get(), "%llu ", &p) == 1);
        assert(ok);
        pVect.push_back(p);
      }
      kVect.push_back(make_pair(k, u32(pVect.size())));
    }
  }

  size_t size() { return kVect.size(); }

  vector<u64> getCovered(u32 pos) {
    auto [k, e] = kVect[pos];
    u32 b = (pos > 0u) ? kVect[pos - 1].second : 0;
    return vector<u64>(pVect.begin() + b, pVect.begin() + e);
  }

  pair<u32, u32> get(u32 pos) {
    assert(pos <= kVect.size());
    if (pos >= kVect.size()) { return make_pair(0xffffffffu, 0u); }
    auto [k, e] = kVect[pos];
    u32 b = (pos > 0u) ? kVect[pos - 1].second : 0;
    return make_pair(k, e - b);
  }
};

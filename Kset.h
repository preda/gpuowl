// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "file.h"
#include "common.h"

#include <cassert>
#include <vector>

class Kset {
  vector<u32> Ks;

public:
  Kset(const string &name) {
    auto fi = openRead(name, true);
    u32 k;
    while (fscanf(fi.get(), "%u", &k) == 1) { Ks.push_back(k); }
  }

  u32 get(u32 pos) {
    assert(pos <= Ks.size());
    return (pos < Ks.size()) ? Ks[pos] : 0xffffffffu;    
  }
};

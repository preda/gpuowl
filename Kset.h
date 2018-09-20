// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"

#include <vector>
#include <string>

class Kset {
  vector<u32> Ks;
  u32 B1;
  
public:
  Kset(const string &name);

  u32 getB1() { return B1; }
  
  u32 get(u32 pos);
};

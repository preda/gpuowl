// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"

#include <vector>
#include <string>

class Kset {
  vector<u32> Ks;
  vector<u32>::iterator hint;
  
public:
  Kset(const string &name);

  u32 getFirstAfter(u32 k);
};

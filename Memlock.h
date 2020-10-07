// Copyright Mihai Preda.

#pragma once

#include "common.h"
#include <filesystem>

namespace fs = std::filesystem;

class Memlock {
  fs::path lock;
  
public:
  Memlock(fs::path base, u32 device);
  ~Memlock();
};

// Copyright (C) Mihai Preda

#pragma once

#include "common.h"

#include <filesystem>
#include <queue>

class File;

class SaveMan {
private:
  std::string kind;
  fs::path base;

  u32 lastK = 0;

  using T = pair<float, u32>;
  using Heap = priority_queue<T, std::vector<T>, std::greater<T>>;
  Heap minVal;

  float value(u32 k);
  void del(u32 k);

  vector<u32> listIterations();
  fs::path path(u32 k);

public:
  const u32 exponent;

  SaveMan(std::string_view kind, u32 exponent);

  File write(u32 k);
  File readLast();


  void removeAll();
  u32 getLastK() const { return lastK; }
};


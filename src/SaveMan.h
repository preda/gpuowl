// Copyright (C) Mihai Preda

#pragma once

#include "common.h"

#include <filesystem>

class File;

class SaveMan {
private:
  std::string kind;
  fs::path base;
  vector<u32> points;

  void del(u32 k);
  vector<u32> listIterations();
  fs::path path(u32 k);

public:
  const u32 exponent;

  SaveMan(std::string_view kind, u32 exponent);

  File write(u32 k);
  File readLast();

  void removeAll();
  u32 getLastK() const { return points.empty() ? 0 : points.back(); }
};

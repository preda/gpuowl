// Copyright (C) Mihai Preda

#include "common.h"

#include <filesystem>

inline fs::path operator+(fs::path p, const std::string& tail) {
  p += tail;
  return p;
}

u64 fileSize(const fs::path& path);

void fancyRename(const fs::path& src, const fs::path& dst);

bool deleteLine(const fs::path& path, const string& targetLine, u64 initialSize = 0);

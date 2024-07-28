// Copyright (C) Mihai Preda

#pragma once

#include "common.h"

#include <filesystem>
#include <optional>

class File;
class SaveMan;

struct PRPState {
  static const constexpr char* KIND = "prp";

  u32 exponent;
  u32 k;
  u32 blockSize;
  u64 res64;
  vector<u32> check;
  u32 nErrors;
};

struct LLState {
  static const constexpr char* KIND = "ll";

  u32 exponent;
  u32 k;
  vector<u32> data;
};

template<typename State>
class Saver {
  u32 exponent;
  u32 blockSize;
  fs::path base;
  fs::path trash;
  string prefix;
  u32 nSavefiles;

  State initState();
  void moveToTrash(fs::path file);
  void trimFiles();
  fs::path mostRecentSavefile();

public:
  Saver(u32 exponent, u32 blockSize, u32 nSavefiles);
  ~Saver();

  State load();
  void save(const State& s);

  void dropMostRecent();

  void clear();

  // For PRP, we can save a verified save (see save() above) or an unverified save.
  void saveUnverified(const PRPState& s) const;
};

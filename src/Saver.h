// Copyright (C) Mihai Preda

#pragma once

#include "SaveMan.h"
#include "common.h"

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
  SaveMan man;
  u32 exponent;
  u32 blockSize;

public:
  Saver(u32 exponent, u32 blockSize);
  ~Saver();

  State load();
  void save(const State& s);
  void clear();

  // For PRP, we can save a verified save (see save() above) or an unverified save.
  void unverifiedSave(const PRPState& s) const;
  PRPState unverifiedLoad();
  void dropUnverified();
};

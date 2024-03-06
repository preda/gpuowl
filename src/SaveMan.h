// Copyright (C) Mihai Preda

#pragma once

#include "common.h"
#include "File.h"

#include <filesystem>
#include <queue>

class FileMan {
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

  FileMan(std::string_view kind, u32 exponent);

  File write(u32 k);
  File readLast();

  void removeAll();
  u32 getLastK() const { return lastK; }
};

struct PRPState;
struct LLState;

template<typename State>
State load(File&& f, u32 E, u32 k);

template<> PRPState load<PRPState>(File&& f, u32 E, u32 k);
template<> LLState load<LLState>(File&& f, u32 E, u32 k);

void save(File&& f, const PRPState& s);
void save(File&& f, const LLState& s);

template<typename State>
class StateSaver {
  FileMan man;

public:
  StateSaver(u32 exponent) :
    man{State::KIND, exponent}
  {}

  State load() {
    return ::load<State>(man.readLast(), man.exponent, man.getLastK());
  }

  void save(const State& s) {
    ::save(man.write(s.k), s);
  }

  void clear() { man.removeAll(); }
};

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

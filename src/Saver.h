// Copyright (C) Mihai Preda.

#pragma once

#include "common.h"

#include <vector>
#include <string>
#include <cinttypes>
#include <queue>
#include <filesystem>

class Args;

struct PRPState {
  u32 k{};
  u32 blockSize{};
  u64 res64{};
  vector<u32> check;
  u32 nErrors{};
};

class Saver {
  const u32 E;
  const u32 nKeep = 20;
  const fs::path base = fs::current_path() / to_string(E);

  // E, k, block-size, res64, nErrors
  static constexpr const char *PRP_v10 = "OWL PRP 10 %u %u %u %016" SCNx64 " %u\n";

  // Exponent, iteration, block-size, res64, nErrors
  // B1, nBits, start, nextK, crc
  static constexpr const char *PRP_v11 = "OWL PRP 11 %u %u %u %016" SCNx64 " %u %u %u %u %u %u\n";

  // E, k, block-size, res64, nErrors, CRC
  static constexpr const char *PRP_v12 = "OWL PRP 12 %u %u %u %016" SCNx64 " %u %u\n";

  static constexpr const char *P1_v3 = "OWL P1 3 E=%u B1=%u k=%u\n";

  // E, k, CRC
  static constexpr const char *LL_v1 = "OWL LL 1 E=%u k=%u CRC=%u\n";

  // ----

  u32 lastK = 0;
  
  using T = pair<float, u32>;
  using Heap = priority_queue<T, std::vector<T>, std::greater<T>>;
  Heap minValPRP;
  Heap minValLL;
  
  float value(u32 k);
  void del(u32 k);

  static string str9(u32 k) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%09u", k);
    return buf;
  }

  fs::path path(u32 k, const string& ext) const {
    return path(to_string(k), ext);
  }

  fs::path path(const string& sk, const string& ext) const {
    return base / (to_string(E) + '-' + sk + ext);
  }

  fs::path pathPRP(u32 k) const { return path(str9(k), ".prp"); }
  fs::path pathLL(u32 k) const { return path(str9(k), ".ll"); }

  void savedPRP(u32 k);

  PRPState loadPRPAux(u32 k);

  vector<u32> listIterations(const string& ext);
  vector<u32> listIterationsPRP() { return listIterations(".prp"); }
  vector<u32> listIterationsLL() { return listIterations(".ll"); }

  void scan(u32 upToK = u32(-1));
  
  
public:
  static void cycle(const fs::path& name);
  static void cleanup(u32 E, const Args& args);
  
  Saver(u32 E);

  PRPState loadPRP(u32 iniBlockSize);  
  void savePRP(const PRPState& state);

  PRPState loadLL();
  void saveLL(const PRPState& state);
};

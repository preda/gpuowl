// GpuOwl Mersenne primality tester; Copyright (C) Mihai Preda.

#pragma once

#include "File.h"
#include "common.h"

#include <vector>
#include <string>
#include <cinttypes>
#include <queue>

class Args;

struct PRPState {
  u32 k{};
  u32 blockSize{};
  u64 res64{};
  vector<u32> check;
  u32 nErrors{};
};

struct P1State {
  u32 B1;
  u32 k;
  u32 blockSize;
  Words data;
};

class Saver {
  // E, k, block-size, res64, nErrors
  static constexpr const char *PRP_v10 = "OWL PRP 10 %u %u %u %016" SCNx64 " %u\n";

  // Exponent, iteration, block-size, res64, nErrors
  // B1, nBits, start, nextK, crc
  static constexpr const char *PRP_v11 = "OWL PRP 11 %u %u %u %016" SCNx64 " %u %u %u %u %u %u\n";

  // E, k, block-size, res64, nErrors, CRC
  static constexpr const char *PRP_v12 = "OWL PRP 12 %u %u %u %016" SCNx64 " %u %u\n";

  static constexpr const char *P1_v3 = "OWL P1 3 E=%u B1=%u k=%u block=%u\n";

  // ----

  u32 lastK = 0;
  
  using T = pair<float, u32>;
  using Heap = priority_queue<T, std::vector<T>, std::greater<T>>;
  Heap minValPRP;
  

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
  fs::path pathP1() const       { return base / to_string(E) + ".p1"; }

  void savedPRP(u32 k);

  PRPState loadPRPAux(u32 k);
  vector<u32> listIterations(const string& prefix, const string& ext);
  vector<u32> listIterations();
  void scan(u32 upToK = u32(-1));
  
  const u32 E;
  const fs::path base = fs::current_path() / to_string(E);
  const u32 nKeep;
  
public:
  static void cycle(const fs::path& name);
  static void cleanup(u32 E, const Args& args);
  
  Saver(u32 E, u32 nKeep, u32 startFrom);


  PRPState loadPRP(u32 iniBlockSize);  
  void savePRP(const PRPState& state);

  P1State loadP1();
  void saveP1(const P1State& state);

  // Will delete all PRP & P-1 savefiles at iteration kBad up to currentK as bad.
  void deleteBadSavefiles(u32 kBad, u32 currentK);
};

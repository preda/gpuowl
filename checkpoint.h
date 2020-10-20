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


using P1State = pair<u32, Words>; // nextK, data

class Saver {
  // E, k, block-size, res64, nErrors
  static constexpr const char *PRP_v10 = "OWL PRP 10 %u %u %u %016" SCNx64 " %u\n";

  // Exponent, iteration, block-size, res64, nErrors
  // B1, nBits, start, nextK, crc
  static constexpr const char *PRP_v11 = "OWL PRP 11 %u %u %u %016" SCNx64 " %u %u %u %u %u %u\n";

  // E, k, block-size, res64, nErrors, CRC
  static constexpr const char *PRP_v12 = "OWL PRP 12 %u %u %u %016" SCNx64 " %u %u\n";

  // E, B1, k, nextK, CRC
  static constexpr const char *P1_v2 = "OWL P1 2 %u %u %u %u %u\n";

  // E, B1, CRC
  static constexpr const char *P1Final_v1 = "OWL P1F 1 %u %u %u\n";

  // E, B1, B2
  static constexpr const char *P2_v2 = "OWL P2 2 %u %u %u\n";
  
  // E, B1, B2, D, nBuf, nextBlock
  static constexpr const char *P2_v3 = "OWL P2 3 %u %u %u %u %u %u\n";

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

  fs::path makePath(const string& prefix, u32 k, const string& ext) const {
    return base / (prefix + '-' + str9(k) + ext);
  }

  fs::path pathPRP(u32 k) const         { return makePath(to_string(E), k, ".prp"); }
  fs::path pathP1(u32 k) const  { return makePath(to_string(E) + '-' + to_string(b1), k, ".p1"); }
  fs::path pathP1Final() const  { return base / (to_string(E) + '-' + to_string(b1) + ".p1final"); }
  fs::path pathP2() const { return base / (to_string(E) + '-' + to_string(b1) + ".p2"); }

  void savedPRP(u32 k);

  PRPState loadPRPAux(u32 k);
  vector<u32> listIterations(const string& prefix, const string& ext);
  vector<u32> listIterations();
  void scan(u32 upToK = u32(-1));
  
  const u32 E;
  const fs::path base = fs::current_path() / to_string(E);
  const u32 nKeep;
  
public:
  const u32 b1;

  
  Saver(u32 E, u32 nKeep, u32 b1, u32 startFrom);

  static void cleanup(u32 E, const Args& args);

  PRPState loadPRP(u32 iniBlockSize);  
  void savePRP(const PRPState& state);

  P1State loadP1(u32 k);
  void saveP1(u32 k, const P1State& state);

  vector<u32> loadP1Final();
  void saveP1Final(const vector<u32>& data);
  
  u32 loadP2(u32 b2, u32 D, u32 nBuf);
  void saveP2(u32 b2, u32 D, u32 nBuf, u32 nextBlock);

  // Will delete all PRP & P-1 savefiles at iteration kBad up to currentK as bad.
  void deleteBadSavefiles(u32 kBad, u32 currentK);
};

#pragma once

#include "common.h"

#include <vector>
#include <string>
#include <cassert>

std::string fileName(int E, const string &suffix = "") { return std::to_string(E) + suffix + ".owl"; }

struct PRPState_v5 {
    static constexpr const char *HEADER_R = R"(OWL 5
Comment: %255[^
]
Type: PRP
Exponent: %u
Iteration: %u
PRP-block-size: %u
Residue-64: 0x%016llx
Errors: %d
End-of-header:
\0)";

  static constexpr const char *SUFFIX = "";
  
  u32 k;
  u32 blockSize;
  int nErrors;
  u64 res64;
  vector<u32> check;
  
  bool load(u32 E) {
    u32 nWords = (E - 1)/32 + 1;
    u32 nBytes = (E - 1)/8 + 1;
    
    auto fi{open(fileName(E, SUFFIX), "rb", false)};
    if (!fi) { return false; }
    
    char buf[256];
    u32 fileE;
    if (!(fscanf(fi.get(), HEADER_R, buf, &fileE, &k, &blockSize, &res64, &nErrors) == 6)) { return false; }
    assert(E == fileE);
    check = vector<u32>(nWords);
    if (!fread(check.data(), nBytes, 1, fi.get())) { return false; }
    return true;
  }
};
  
template<typename State>
bool save(u32 E, State *state) {
  string tempFile = fileName(E, "-temp"s + State::SUFFIX);
  if (!state->saveImpl(E, tempFile)) { return false; }
  
  string prevFile = fileName(E, "-prev"s + State::SUFFIX);
  remove(prevFile.c_str());

  string saveFile = fileName(E, State::SUFFIX);
  rename(saveFile.c_str(), prevFile.c_str());
  rename(tempFile.c_str(), saveFile.c_str());

  string persist = state->durableName();
  return persist.empty() ? true : state->saveImpl(E, fileName(E, persist + State::SUFFIX));
}

bool write(FILE *fo, const vector<u32> &v) { return fwrite(v.data(), v.size() * sizeof(u32), 1, fo); }
bool read(FILE *fi, u32 nWords, vector<u32> *v) {
  v->resize(nWords);
  return fread(v->data(), nWords * sizeof(u32), 1, fi);
}

struct PRPState {
public:
  // Exponent, iteration, block-size, res64, nErrors.
  static constexpr const char *HEADER = "OWL PRP 6 %u %u %u %016llx %u\n";
  static constexpr const char *SUFFIX = ".prp";

  bool load_v5(u32 E) {
    PRPState_v5 v5;
    if (!v5.load(E)) { return false; }

    k = v5.k;
    blockSize = v5.blockSize;
    nErrors = v5.nErrors;
    res64 = v5.res64;
    check = move(v5.check);
    return true;
  }
  
public:
  u32 k;
  u32 blockSize;
  u32 nErrors;
  u64 res64;
  vector<u32> check;

  bool saveImpl(u32 E, const string &name) {
    assert(check.size() == (E - 1) / 32 + 1);
    auto fo(open(name, "wb"));
    return fo
      && fprintf(fo.get(),  HEADER, E, k, blockSize, res64, nErrors) > 0
      && write(fo.get(), check);
  }

  string durableName() { return k && (k % 20'000'000 == 0) ? "."s + to_string(k/1'000'000)+"M" : ""s; }

  bool save(u32 E) { return ::save(E, this); }
  
  bool load(u32 E, u32 iniBlockSize) {    
    u32 nWords = (E - 1) / 32 + 1;
    auto fi{open(fileName(E, SUFFIX), "rb", false)};
    if (!fi) {
      if (load_v5(E)) { return true; }

      k = 0;
      blockSize = iniBlockSize;
      nErrors = 0;
      res64 = 3;
      check = vector<u32>(nWords);
      check[0] = 1;
      return true;
    }
    char line[256];
    u32 fileE;    
    if (fgets(line, sizeof(line), fi.get())
        && sscanf(line, HEADER, &fileE, &k, &blockSize, &res64, &nErrors) == 5
        && read(fi.get(), nWords, &check)) {
      assert(E == fileE);
      return true;
    }
    return false;
  }
};

struct PFState {
  // Exponent, iteration, total-iterations, B1.
  static constexpr const char *HEADER = "OWL PF 1 %u %u %u %u\n";
  static constexpr const char *SUFFIX = ".pf";

  u32 k, kEnd, B1;
  vector<u32> base;

  bool saveImpl(u32 E, const string &name) {
    assert(base.size() == (E - 1) / 32 + 1);
    
    auto fo(open(name, "wb"));
    return fo
      && fprintf(fo.get(), HEADER, E, k, kEnd, B1) > 0
      && write(fo.get(), base);
  }

  string durableName() { return ""; }
  
  bool save(u32 E) { return ::save(E, this); }
  
  bool load(u32 E, u32 iniB1) {
    u32 nWords = (E - 1)/32 + 1;
    
    auto fi{open(fileName(E, SUFFIX), "rb", false)};
    if (!fi) {
      k = 0;
      kEnd = 0; // not known yet.
      B1 = iniB1;
      base = vector<u32>(nWords);
      base[0] = 1;
      return true;
    }

    char line[256];
    u32 fileE;
    if (!fgets(line, sizeof(line), fi.get())
        || !(sscanf(line, HEADER, &fileE, &k, &kEnd, &B1) == 4)
        || !read(fi.get(), nWords, &base)) {
      return false;
    }
    assert(E == fileE);
    return true;
  }

  bool finished() {
    assert(k <= kEnd);
    return k >= kEnd;
  }
};

struct PRPFState {
public:
  // E, k, B1, blockSize, res64
  static constexpr const char *HEADER = "OWL PRPF 1 %u %u %u %u %016llx\n";
  static constexpr const char *SUFFIX = ".prpf";

public:
  u32 k;
  u32 B1;
  u32 blockSize;
  u64 res64;
  vector<u32> base;
  vector<u32> check;

  bool saveImpl(u32 E, string name) {
    u32 nWords = (E - 1) / 32 + 1;
    assert(base.size() == nWords && check.size() == nWords);

    auto fo(open(name, "wb"));
    return fo
      && fprintf(fo.get(), HEADER, E, k, B1, blockSize, res64) == 5
      && write(fo.get(), base)
      && write(fo.get(), check);    
  }

  string durableName() { return (k % 20'000'000 == 0) ? "."s + to_string(k/1000000)+"M":""s; }
  
  bool save(u32 E) { return ::save(E, this); }
  
  bool load(u32 E, u32 iniB1, u32 iniBlockSize) {
    u32 nWords = (E - 1) / 32 + 1;
    
    auto fi{open(fileName(E, SUFFIX), "rb", false)};
    if (!fi) {
      PFState pf;
      if (!pf.load(E, iniB1) || !pf.finished()) { return false; }
      k = 0;
      B1 = pf.B1;
      blockSize = iniBlockSize;
      base = move(pf.base);
      assert(base.size() == nWords);
      res64 = (u64(base[1]) << 32) | base[0];
      check = vector<u32>(nWords);
      check[0] = 1;
      return true;
    }

    char line[256];
    u32 fileE;
    if (fgets(line, sizeof(line), fi.get())
        && sscanf(line, HEADER, &fileE, &k, &B1, &blockSize, &res64) == 5
        && read(fi.get(), nWords, &base)
        && read(fi.get(), nWords, &check)) {
      assert(E == fileE);
      return true;
    }
    return false;
  }
};

struct TFState {
  // Exponent, bitLo, bitHi, classDone, classTotal.
  static constexpr const char *HEADER = "OWL TF 2 %u %u %u %u %u\n";
  static constexpr const char *SUFFIX = ".tf";
  u32 bitLo, bitHi;
  u32 nDone, nTotal;

  bool save(u32 E) { return ::save(E, this); }
  string durableName() { return ""; }

  bool saveImpl(u32 E, string name) {
    auto fo(open(name, "wb"));
    return fo && fprintf(fo.get(), HEADER, E, bitLo, bitHi, nDone, nTotal) > 0;
  }

  bool load(u32 E) {
    auto fi(open(fileName(E, SUFFIX), "rb", false));
    if (!fi) {
      bitLo = bitHi = nDone = nTotal = 0;
      return true;
    }

    u32 fileE;
    if (fscanf(fi.get(), HEADER, &fileE, &bitLo, &bitHi, &nDone, &nTotal) == 5) {
      assert(E == fileE);
      assert(bitLo < bitHi);
      return true;
    }
    return false;
  }
};

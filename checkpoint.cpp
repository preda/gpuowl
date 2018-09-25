// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "checkpoint.h"
#include "file.h"
#include <cassert>

static std::string fileName(int E, const string &suffix) { return std::to_string(E) + suffix + ".owl"; }

template<typename T> void save(u32 E, T *state) {
  string tempFile = fileName(E, "-temp"s + T::SUFFIX);
  if (!state->saveImpl(E, tempFile)) {
    throw "can't save";
  }
  
  string prevFile = fileName(E, "-prev"s + T::SUFFIX);
  remove(prevFile.c_str());
  
  string saveFile = fileName(E, T::SUFFIX);
  rename(saveFile.c_str(), prevFile.c_str());
  rename(tempFile.c_str(), saveFile.c_str());
  
  string persist = state->durableName();
  if (!persist.empty() && !state->saveImpl(E, fileName(E, persist + T::SUFFIX))) {
    throw "can't save";
  }
}

template void save<PRPState>(u32, PRPState *);
template void save<TFState>(u32, TFState *);

static bool write(FILE *fo, const vector<u32> &v) {
  return fwrite(v.data(), v.size() * sizeof(u32), 1, fo);
}

static bool read(FILE *fi, u32 nWords, vector<u32> *v) {
  v->resize(nWords);
  return fread(v->data(), nWords * sizeof(u32), 1, fi);
}

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

  u32 k;
  u32 blockSize;
  int nErrors;
  u64 res64;
  vector<u32> check;
  
  bool load(u32 E) {
    u32 nWords = (E - 1)/32 + 1;
    u32 nBytes = (E - 1)/8 + 1;
    
    auto fi{openRead(fileName(E, ""))};
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

bool PRPState::load_v5(u32 E) {
  PRPState_v5 v5;
  if (!v5.load(E)) { return false; }
  
  k = v5.k;
  blockSize = v5.blockSize;
  // nErrors = v5.nErrors;
  res64 = v5.res64;
  check = move(v5.check);
  return true;
}

bool PRPState::load_v6(u32 E) {
  const char *HEADER = "OWL PRP 6 %u %u %u %016llx %u\n";
  u32 nWords = (E - 1) / 32 + 1;
  if (auto fi{openRead(fileName(E, ".prp"))}) {
    char line[256];
    u32 fileE;
    u32 nErrors;
    bool ok = fgets(line, sizeof(line), fi.get())
      && sscanf(line, HEADER, &fileE, &k, &blockSize, &res64, &nErrors) == 5
      && read(fi.get(), nWords, &check);
    if (ok) {
      assert(E == fileE);
      return true;
    }
  }
  return load_v5(E);
}

void PRPState::loadInt(u32 E, u32 wantB1, u32 iniBlockSize) {
  u32 nWords = (E - 1) / 32 + 1;
  string name = fileName(E, SUFFIX);
  if (auto fi{openRead(name)}) {
    char line[256];
    u32 fileE;
    bool ok = fgets(line, sizeof(line), fi.get())
      && sscanf(line, HEADER, &fileE, &k, &B1, &blockSize, &res64) == 5
      && read(fi.get(), nWords, &check);
    if (ok) {
      assert(E == fileE);
      if (B1 != wantB1) {
        log("B1 mismatch: using B1=%u from '%s' instead of %u\n", B1, name.c_str(), wantB1);
      }
      
      if (B1 == 0) {
        base = vector<u32>(nWords);
        base[0] = 3;
      } else {
        bool ok = read(fi.get(), nWords, &base);
        assert(ok);
      }
      return;
    }
  }
  
  if (load_v6(E)) {
    if (wantB1 != 0) { log("B1 mismatch: using B1=0 from from savefile\n"); }
    base = vector<u32>(nWords);
    base[0] = 3;
    return;
  }

  log("PRP savefile not found '%s'\n", fileName(E, SUFFIX).c_str());
  assert(false);
}

bool PRPState::saveImpl(u32 E, const string &name) {
  assert(check.size() == (E - 1) / 32 + 1);
  auto fo(openWrite(name));
  return fo
    && fprintf(fo.get(),  HEADER, E, k, B1, blockSize, res64) > 0
    && write(fo.get(), check)
    && write(fo.get(), base);
}

string PRPState::durableName() {
  return k && (k % 20'000'000 == 0) ? "."s + to_string(k/1'000'000)+"M" : ""s;
}

bool PRPState::exists(u32 E) { return openRead(fileName(E, SUFFIX)) || openRead(fileName(E,".prp")); }

void TFState::loadInt(u32 E) {    
  if (auto fi{openRead(fileName(E, SUFFIX))}) {
    u32 fileE;
    if (fscanf(fi.get(), HEADER, &fileE, &bitLo, &bitHi, &nDone, &nTotal) == 5) {
      assert(E == fileE);
      assert(bitLo < bitHi);
    } else {
      log("Can't parse file '%s'\n", fileName(E, SUFFIX).c_str());
      throw "parse";
    }
  } else {
    bitLo = bitHi = nDone = nTotal = 0;
  }
}

bool TFState::saveImpl(u32 E, string name) {
  auto fo(openWrite(name));
  return fo && fprintf(fo.get(), HEADER, E, bitLo, bitHi, nDone, nTotal) > 0;
}

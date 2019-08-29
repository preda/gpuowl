// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "checkpoint.h"
#include "file.h"

#include <cassert>
#include <cmath>
#include <gmp.h>

// Residue from compacted words.
u64 residue(const vector<u32> &words) { return (u64(words[1]) << 32) | words[0]; }

static std::string fileName(u32 E, const string &suffix) { return std::to_string(E) + suffix + ".owl"; }

void PRPState::save(u32 E) {
  string tempFile = fileName(E, "-temp"s + SUFFIX);
  if (!saveImpl(E, tempFile)) {
    throw "can't save";
  }
  
  string prevFile = fileName(E, "-prev"s + SUFFIX);
  remove(prevFile.c_str());
  
  string saveFile = fileName(E, SUFFIX);
  rename(saveFile.c_str(), prevFile.c_str());
  rename(tempFile.c_str(), saveFile.c_str());
  
  string persist = durableName();
  if (!persist.empty() && !saveImpl(E, fileName(E, persist + SUFFIX))) {
    throw "can't save";
  }
}

static bool write(FILE *fo, const vector<u32> &v) {
  return fwrite(v.data(), v.size() * sizeof(u32), 1, fo);
}

static bool read(FILE *fi, u32 nWords, vector<u32> *v) {
  v->resize(nWords);
  return fread(v->data(), nWords * sizeof(u32), 1, fi);
}

static vector<u32> makeVect(u32 size, u32 elem0) {
  vector<u32> v(size);
  v[0] = elem0;
  return v;
}

void PRPState::loadInt(u32 E, u32 iniBlockSize) {
  u32 nWords = (E - 1) / 32 + 1;
  string name = fileName(E, SUFFIX);
  auto fi{openRead(name)};
  if (!fi) {
    log("%s not found, starting from the beginning.\n", name.c_str());
    k = 0;
    blockSize = iniBlockSize;
    res64  = 3;
    check = makeVect(nWords, 1);
    return;
  }
  
  char line[256];
  if (!fgets(line, sizeof(line), fi.get())) {
    log("Invalid savefile '%s'\n", name.c_str());
    throw("invalid savefile");
  }
  
  u32 fileE = 0;
  if (sscanf(line, HEADER_v9, &fileE, &k, &blockSize, &res64) == 4) {
    assert(E == fileE);
    if (!read(fi.get(), nWords, &check)) { throw("load: error read check"); }
    #ifdef __MINGW64__
      log("%s loaded: k %u, block %u, res64 %016I64x\n", name.c_str(), k, blockSize, res64);
    #else
      log("%s loaded: k %u, block %u, res64 %016llx\n", name.c_str(), k, blockSize, res64);
    #endif
  } else {
    log("Invalid savefile '%s'\n", name.c_str());
    throw("invalid savefile");    
  }
}

bool PRPState::saveImpl(u32 E, const string &name) {
  u32 nWords = (E - 1) / 32 + 1;
  assert(check.size() == nWords);

  auto fo(openWrite(name));
  return
    fo
    && fprintf(fo.get(), HEADER_v9, E, k, blockSize, res64) > 0
    && write(fo.get(), check);
}

string PRPState::durableName() {
  if (k && (k % 20'000'000 == 0)) { return "."s + to_string(k/1'000'000)+"M"; }
  return "";
}

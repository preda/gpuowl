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

static bool write(FILE *fo, const vector<u32> &v) {
  return fwrite(v.data(), v.size() * sizeof(u32), 1, fo);
}

static bool read(FILE *fi, u32 nWords, vector<u32> *v) {
  v->resize(nWords);
  return fread(v->data(), nWords * sizeof(u32), 1, fi);
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

bool PRPState::exists(u32 E) { return bool(openRead(fileName(E, SUFFIX))); }

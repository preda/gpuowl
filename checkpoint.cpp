// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "checkpoint.h"
#include "file.h"

#include <filesystem>
#include <ios>
#include <cassert>
#include <cmath>
#include <gmp.h>

namespace fs = std::filesystem;

// Residue from compacted words.
u64 residue(const vector<u32> &words) { return (u64(words[1]) << 32) | words[0]; }

static std::string fileName(u32 E, const string &suffix) { return std::to_string(E) + suffix + ".owl"; }

void PRPState::save() {
  string newFile = fileName(E, "-new"s + SUFFIX);
  saveImpl(newFile);
  
  string saveFile = fileName(E, SUFFIX);
  fs::rename(saveFile, fileName(E, "-old"s + SUFFIX));
  fs::rename(newFile, saveFile);
}

static void write(FILE *fo, const vector<u32> &v) {
  if (!fwrite(v.data(), v.size() * sizeof(u32), 1, fo)) { throw(std::ios_base::failure("can't write data")); }
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

PRPState::PRPState(u32 E, u32 iniBlockSize)
  : E(E) {
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
    log("%s loaded: k %u, block %u, res64 %s\n", name.c_str(), k, blockSize, hex(res64).c_str());
  } else {
    log("Invalid savefile '%s'\n", name.c_str());
    throw("invalid savefile");    
  }
}

void PRPState::saveImpl(const string &name) {
  u32 nWords = (E - 1) / 32 + 1;
  assert(check.size() == nWords);

  auto fo{openWrite(name)};
  if (fprintf(fo.get(), HEADER_v9, E, k, blockSize, res64) <= 0) { throw(ios_base::failure("can't write header")); }
  write(fo.get(), check);
}

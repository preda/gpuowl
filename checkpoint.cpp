// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "checkpoint.h"
#include "file.h"

#include <filesystem>
#include <ios>
#include <cassert>

namespace fs = std::filesystem;

// Residue from compacted words.
u64 residue(const vector<u32> &words) { return (u64(words[1]) << 32) | words[0]; }

static fs::path fileName(u32 E, const string &suffix) {
  string sE = to_string(E);
  auto baseDir = fs::current_path() / sE;
  if (!fs::exists(baseDir)) { fs::create_directory(baseDir); }
  return baseDir / (sE + suffix + ".owl");
}

void PRPState::save() {
  fs::path newFile = fileName(E, "-new"s + SUFFIX);
  saveImpl(newFile.string());
  
  fs::path saveFile = fileName(E, SUFFIX);
  fs::path oldFile = fileName(E, "-old"s + SUFFIX);
  error_code noThrow;
  fs::remove(oldFile);
  fs::rename(saveFile, oldFile, noThrow);
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

bool PRPState::load(FILE* fi) {
  char line[256];
  if (!fgets(line, sizeof(line), fi)) {
    // throw("invalid savefile");
    return false;
  }
  
  u32 fileE = 0;
  if (sscanf(line, HEADER_v9, &fileE, &k, &blockSize, &res64) == 4) {
    assert(E == fileE);
    u32 nWords = (E - 1) / 32 + 1;
    return read(fi, nWords, &check);
  } else {
    return false;
    // throw("invalid savefile");    
  }
}

PRPState::PRPState(u32 E, u32 iniBlockSize)
  : E(E) {
  bool foundFiles = false;
  // unique_ptr<FILE> fi;
  for (auto&& path : {fileName(E, SUFFIX), fileName(E, "-old"s + SUFFIX)}) {    
    if (auto fi = openRead(path.string())) {
      foundFiles = true;
      if (load(fi.get())) {
        log("%s loaded: k %u, block %u, res64 %s\n", path.string().c_str(), k, blockSize, hex(res64).c_str());
        return;
      } else {
        log("Invalid savefile '%s'\n", path.string().c_str());
      }
    } else {
      log("%s not found\n", path.string().c_str());
    }
  }
  
  if (foundFiles) {
    log("invalid savefiles found, investigate why\n");
  } else {
    log("starting from the beginning.\n");
    k = 0;
    blockSize = iniBlockSize;
    res64  = 3;
    u32 nWords = (E - 1) / 32 + 1;
    check = makeVect(nWords, 1);
  }
}

void PRPState::saveImpl(const string &name) {
  u32 nWords = (E - 1) / 32 + 1;
  assert(check.size() == nWords);

  auto fo{openWrite(name)};
  if (fprintf(fo.get(), HEADER_v9, E, k, blockSize, res64) <= 0) { throw(ios_base::failure("can't write header")); }
  write(fo.get(), check);
}

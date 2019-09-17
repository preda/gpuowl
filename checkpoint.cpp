// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "checkpoint.h"
#include "file.h"

#include <filesystem>
#include <ios>
#include <cassert>

namespace fs = std::filesystem;

// Residue from compacted words.
u64 residue(const vector<u32> &words) { return (u64(words[1]) << 32) | words[0]; }

static fs::path fileName(u32 E, const string& suffix = "", const string& extension = "owl") {
  string sE = to_string(E);
  auto baseDir = fs::current_path() / sE;
  if (!fs::exists(baseDir)) { fs::create_directory(baseDir); }
  return baseDir / (sE + suffix + '.' + extension);
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

void StateLoader::save(u32 E, const std::string& extension) {
  fs::path newFile = fileName(E, "-new", extension);
  doSave(openWrite(newFile.string()).get());
  
  fs::path saveFile = fileName(E, "", extension);
  fs::path oldFile = fileName(E, "-old", extension);
  error_code noThrow;
  fs::remove(oldFile, noThrow);
  fs::rename(saveFile, oldFile, noThrow);
  fs::rename(newFile, saveFile);
}

bool StateLoader::load(u32 E, const std::string& extension) {
  bool foundFiles = false;
  for (auto&& path : {fileName(E, "", extension), fileName(E, "-old", extension)}) {
    if (auto fi = openRead(path.string())) {
      foundFiles = true;
      if (load(fi.get())) {
        log("'%s' loaded\n", path.string().c_str());
        return true;
      } else {
        log("'%s' invalid\n", path.string().c_str());
      }
    } else {
      log("'%s' not found\n", path.string().c_str());
    }
  }
  
  if (foundFiles) {
    throw("invalid savefiles found, investigate why\n");
  }
  return false;
}

PRPState::PRPState(u32 E, u32 iniBlockSize) : E{E} {
  if (!load(E, "owl")) {  
    log("starting from the beginning.\n");
    k = 0;
    blockSize = iniBlockSize;
    res64  = 3;
    u32 nWords = (E - 1) / 32 + 1;
    check = makeVect(nWords, 1);
  }
}

bool PRPState::doLoad(const char* headerLine, FILE *fi) {
  u32 fileE = 0;
  if (sscanf(headerLine, HEADER_v10, &fileE, &k, &blockSize, &res64, &nErrors) == 5
      || sscanf(headerLine, HEADER_v9, &fileE, &k, &blockSize, &res64) == 4) {
    assert(E == fileE);
    u32 nWords = (E - 1) / 32 + 1;
    return read(fi, nWords, &check);
  } else {
    return false;
  }
}

void PRPState::doSave(FILE* fo) {
  u32 nWords = (E - 1) / 32 + 1;
  assert(check.size() == nWords);
  if (fprintf(fo, HEADER_v10, E, k, blockSize, res64, nErrors) <= 0) { throw(ios_base::failure("can't write header")); }
  write(fo, check);
}


Pm1State::Pm1State(u32 E) : E{E} {
  if (!load(E, "pm1")) {  
    log("starting from the beginning.\n");
    k = 0;
    B1 = 0;
    nBits = 0;
    u32 nWords = (E - 1) / 32 + 1;
    data = makeVect(nWords, 3);
  }
}

bool Pm1State::doLoad(const char* headerLine, FILE *fi) {
  u32 fileE = 0;
  if (sscanf(headerLine, HEADER_v1, &fileE, &B1, &k, &nBits) == 4) {
    assert(E == fileE);
    u32 nWords = (E - 1) / 32 + 1;
    return read(fi, nWords, &data);
  } else {
    return false;
  }
}

void Pm1State::doSave(FILE* fo) {
  u32 nWords = (E - 1) / 32 + 1;
  assert(data.size() == nWords);
  if (fprintf(fo, HEADER_v1, E, B1, k, nBits) <= 0) { throw(ios_base::failure("can't write header")); }
  write(fo, data);
}

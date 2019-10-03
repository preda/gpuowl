// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "checkpoint.h"
#include "File.h"

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

template<typename T>
static void write(FILE *fo, const vector<T> &v) {
  if (!fwrite(v.data(), v.size() * sizeof(T), 1, fo)) { throw(std::ios_base::failure("can't write data")); }
}

template<typename T>
static bool read(FILE *fi, u32 nWords, vector<T> *v) {
  v->resize(nWords);
  return fread(v->data(), nWords * sizeof(T), 1, fi);
}

static vector<u32> makeVect(u32 size, u32 elem0) {
  vector<u32> v(size);
  v[0] = elem0;
  return v;
}

ResidueSet::ResidueSet(u32 E) : E{E}, step_{(E - E % 1024) / 512} {
  fs::path name = fileName(E, "", "set.owl");
  File f{File::openReadAppend(name)};
}

void StateLoader::save(u32 E, const std::string& extension) {
  fs::path newFile = fileName(E, "-new", extension);
  doSave(File::openWrite(newFile).get());
  
  fs::path saveFile = fileName(E, "", extension);
  fs::path oldFile = fileName(E, "-old", extension);
  error_code noThrow;
  fs::remove(oldFile, noThrow);
  fs::rename(saveFile, oldFile, noThrow);
  fs::rename(newFile, saveFile);
  // log("'%s' saved at %u\n", saveFile.string().c_str(), getK());
}

bool StateLoader::load(u32 E, const std::string& extension) {
  bool foundFiles = false;
  for (auto&& path : {fileName(E, "", extension), fileName(E, "-old", extension)}) {
    if (auto fi = File::openRead(path)) {
      foundFiles = true;
      if (load(fi.get())) {
        // log("'%s' loaded at %u\n", path.string().c_str(), getK());
        return true;
      } else {
        log("'%s' invalid\n", path.string().c_str());
      }
    } else {
      // log("'%s' not found\n", path.string().c_str());
    }
  }
  
  if (foundFiles) {
    throw("invalid savefiles found, investigate why\n");
  }
  return false;
}

PRPState::PRPState(u32 E, u32 iniBlockSize) : E{E} {
  if (!load(E, "owl")) {  
    // log("starting from the beginning\n");
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

// --- P1 ---

P1State::P1State(u32 E, u32 B1) : E{E}, B1{B1} {
  if (!load(E, EXT)) {  
    // log("%u P1 starting from the beginning.\n", E);
    k = 0;
    nBits = 0;
    u32 nWords = (E - 1) / 32 + 1;
    data = makeVect(nWords, 1);
  }
}

bool P1State::doLoad(const char* headerLine, FILE *fi) {
  u32 fileE = 0;
  u32 fileB1 = 0;
  if (sscanf(headerLine, HEADER_v1, &fileE, &fileB1, &k, &nBits) == 4) {
    assert(E == fileE);
    if (B1 != fileB1) {
      log("%u P1 wants B1=%u but savefile has B1=%u. Fix B1 or move savefile\n", E, B1, fileB1);
    } else {
      u32 nWords = (E - 1) / 32 + 1;
      return read(fi, nWords, &data);
    }
  }
  return false;
}

void P1State::doSave(FILE* fo) {
  u32 nWords = (E - 1) / 32 + 1;
  assert(data.size() == nWords);
  if (fprintf(fo, HEADER_v1, E, B1, k, nBits) <= 0) { throw(ios_base::failure("can't write header")); }
  write(fo, data);
}

// --- P2 ---

P2State::P2State(u32 E, u32 B1, u32 B2) : E{E}, B1{B1}, B2{B2} {
  if (!load(E, EXT)) {  
    // log("%u P2 starting from the beginning.\n", E);
    k = 0;
    raw.clear();
  }
}

bool P2State::doLoad(const char* headerLine, FILE *fi) {
  u32 fileE = 0;
  u32 fileB1 = 0;
  u32 fileB2 = 0;
  u32 nWords = 0;
  if (sscanf(headerLine, HEADER_v1, &fileE, &fileB1, &fileB2, &nWords, &k) == 5) {
    assert(E == fileE);
    assert(k > 0 && k < 2880);
    if (B1 != fileB1 || B2 != fileB2) {
      log("%u P2 want B1=%u,B2=%u but savefile has B1=%u,B2=%u. Fix B1,B2 or move savefile\n", E, B1, B2, fileB1, fileB2);
    } else {    
      return read(fi, nWords, &raw);
    }
  }
  return false;
}

void P2State::doSave(FILE* fo) {
  if (fprintf(fo, HEADER_v1, E, B1, B2, u32(raw.size()), k) <= 0) { throw(ios_base::failure("can't write header")); }
  write(fo, raw);
}

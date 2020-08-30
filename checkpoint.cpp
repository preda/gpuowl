// GpuOwl Mersenne primality tester; Copyright (C) Mihai Preda.

#include "checkpoint.h"
#include "File.h"
#include "Blake2.h"

#include <filesystem>
#include <ios>
#include <cassert>
#include <cinttypes>

namespace fs = std::filesystem;

namespace {

fs::path fileName(u32 E, u32 k, const string& suffix, const string& extension) {
  string sE = to_string(E);
  auto baseDir = fs::current_path() / sE;
  if (!fs::exists(baseDir)) { fs::create_directory(baseDir); }
  return baseDir / (to_string(k) + suffix + '.' + extension);
}

void cleanup(u32 E, const string& ext) {
  error_code noThrow;
  fs::remove(fileName(E, E, "", ext), noThrow);
  fs::remove(fileName(E, E, "-old", ext), noThrow);
  
  // attempt delete the exponent folder in case it is now empty
  fs::remove(fs::current_path() / to_string(E), noThrow);
}

u32 nWords(u32 E) { return (E - 1) / 32 + 1; }

}

void PRPState::cleanup(u32 E) { ::cleanup(E, EXT); }
void  P1State::cleanup(u32 E) { ::cleanup(E, EXT); }
void  P2State::cleanup(u32 E) { ::cleanup(E, EXT); }


template<typename T>
static void write(FILE *fo, const vector<T> &v) {
  if (!v.empty() && !fwrite(v.data(), v.size() * sizeof(T), 1, fo)) { throw(std::ios_base::failure("can't write data")); }
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

void StateLoader::save(u32 E, const std::string& extension, u32 k) {
  fs::path newFile = fileName(E, E, "-new", extension);
  doSave(File::openWrite(newFile).get());
  
  fs::path saveFile = fileName(E, E, "", extension);
  fs::path oldFile = fileName(E, E, "-old", extension);
  error_code noThrow;
  fs::remove(oldFile, noThrow);
  fs::rename(saveFile, oldFile, noThrow);
  fs::rename(newFile, saveFile);

  if (k) {
    fs::path persistFile = fileName(E, k, "", extension);
    fs::remove(persistFile, noThrow);    
    fs::copy_file(saveFile, persistFile, fs::copy_options::overwrite_existing);
  }
  
  // log("'%s' saved at %u\n", saveFile.string().c_str(), getK());
}

bool StateLoader::load(u32 E, const std::string& extension) {
  bool foundFiles = false;
  for (auto&& path : {fileName(E, E, "", extension), fileName(E, E, "-old", extension)}) {
    if (auto fi = File::openRead(path)) {
      foundFiles = true;
      if (loadFile(fi.get())) {
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


// --- LL ---

LLState::LLState(u32 E) : E{E} {
  if (!load(E, EXT)) {  
    k = 0;
    data = makeVect(nWords(E), 4);
  }
}

bool LLState::doLoad(const char* headerLine, FILE *fi) {
  u32 fileE = 0;
  u64 fileHash = 0;
  if (sscanf(headerLine, HEADER_v1, &fileE, &k, &fileHash) != 3) {
    log("invalid header\n");
    return false;
  }
  assert(E == fileE);
  if (!read(fi, nWords(E), &data)) {
    log("can't read data\n");
    return false;
  }
  u64 hash = Blake2::hash(E, k, data);
  bool hashOK = (hash == fileHash);
  if (!hashOK) {
    // log("Hash %u %u %lu\n", E, k, data.size());
    log("hash mismatch %" PRIx64 " vs. expected %" PRIx64 "\n", hash, fileHash);
  }
  return hashOK;
}

void LLState::doSave(FILE* fo) {
  assert(data.size() == nWords(E));
  u64 hash = Blake2::hash(E, k, data);
  // log("Hash %u %u %lu %lx %lx\n", E, k, data.size(), hash, h2);
  if (fprintf(fo, HEADER_v1, E, k, hash) <= 0) { throw(ios_base::failure("can't write header")); }
  write(fo, data);
}


// --- PRP ---

PRPState::PRPState(u32 E, u32 iniBlockSize) : E{E} {
  if (!load(E, EXT)) {
    // log("starting from the beginning\n");
    k = 0;
    blockSize = iniBlockSize ? iniBlockSize : 500;
    res64  = 3;
    check = makeVect(nWords(E), 1);
  }
}

bool B1State::load(u32 E, FILE *fi) {
  // B1, nBits, nextK, crc
  u32 crc;
  char c;
  if (fscanf(fi, "%u %u %u %u%c", &b1, &nBits, &nextK, &crc, &c) != 5) { return false; }

  if (c != '\n') {
    log("Invalid B1State header\n");
    return false;
  }
  
  if (b1) {
    if (!read(fi, (E-1)/32+1, &data)) {
      log("Can't read B1State residue\n");
      return false;
    }
    assert(crc32(data) == crc);
  }    
  return true;
}

void B1State::save(FILE *fo) {
  if (fprintf(fo, "%u %u %u %u\n", b1, nBits, nextK, crc32(data)) <= 0) {
    throw(ios_base::failure("B1State can't write header"));
  }
  write(fo, data);
}

bool PRPState::doLoad(const char* headerLine, FILE *fi) {
  u32 fileE = 0;
  if (sscanf(headerLine, HEADER_v11, &fileE, &k, &blockSize, &res64, &nErrors) == 5) {
    assert(E == fileE);
    assert(k > 0);
    
    if (!read(fi, nWords(E), &check)) {
      log("Can't read PRP residue\n");
      return false;
    }
    
    return highB1.load(E, fi) && lowB1.load(E, fi);
  } else if (sscanf(headerLine, HEADER_v10, &fileE, &k, &blockSize, &res64, &nErrors) == 5) {
    if (read(fi, nWords(E), &check)) { return true; }
    log("Can't read PRP residue\n");
  }
  // log("Can't parse header '%s'\n", headerLine);
  return false;
}

void PRPState::doSave(FILE* fo) {
  assert(check.size() == nWords(E));
  /*
  assert(b1High > b1Low || (!biHigh && !b1Low));
  assert(!b1High || (nHighBits > k - 1));
  assert(!b1Low  || (nLowBits  > k - 1));
  assert(high.empty() == (b1High == 0));
  assert(low.empty() == (b1Low == 0));
  */

  if (fprintf(fo, HEADER_v11, E, k, blockSize, res64, nErrors) <= 0) {
    throw(ios_base::failure("can't write header"));
  }
    
  write(fo, check);
  highB1.save(fo);
  lowB1.save(fo);
}

// --- P1 ---

P1State::P1State(u32 E, u32 B1) : E{E}, B1{B1} {
  if (!load(E, EXT)) {  
    // log("%u P1 starting from the beginning.\n", E);
    k = 0;
    nBits = 0;
    data = makeVect(nWords(E), 1);
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
      return read(fi, nWords(E), &data);
    }
  }
  return false;
}

void P1State::doSave(FILE* fo) {
  assert(data.size() == nWords(E));
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

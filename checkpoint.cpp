// GpuOwl Mersenne primality tester; Copyright (C) Mihai Preda.

#include "checkpoint.h"
#include "File.h"
#include "Blake2.h"

#include <filesystem>
#include <functional>
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

/*
vector<string> listFiles(const fs::path& path, const string& ext) {
  vector<string> ret;
  for (auto it : fs::directory_iterator(path)) {
    if (it->is_regular_file()) {
      string name = it->path().filename().string();
      if (name.size() >= tail.size() && name.substr(name.size() - tail.size()) == tail) { ret.push_back(name); }
    }
  }
  return ret;
}
*/

vector<u32> listIterations(u32 E, const string& ext) {
  vector<u32> ret;
  fs::path path = fs::current_path() / to_string(E);
  for (auto entry : fs::directory_iterator(path)) {
    if (entry.is_regular_file()) {
      string name = entry.path().filename().string();
      u32 dot = name.find('.');
      if (dot != string::npos && name.substr(dot) == ext) {
        string head = name.substr(0, dot);
        u32 k = std::stoul(head);
        ret.push_back(k);
      }
    }
  }
  std::sort(ret.begin(), ret.end());
  return ret;
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
  doSave(File::openWrite(newFile).syncOnClose().get());
  
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


// --- PRP ---

PRPState PRPState::load(u32 E, u32 iniBlockSize) {
  vector<u32> iterations = listIterations(E, EXT);
  if (iterations.empty()) {
    log("PRP starting from the beginning\n");
    u32 blockSize = iniBlockSize ? iniBlockSize : 500;
    return {0, blockSize, 3, makeVect(nWords(E), 1), 0};
  } else {
    u32 k = iterations.back();
    return loadInt(E, k);
  }
}

PRPState PRPState::loadInt(u32 E, u32 k) {
  assert(k > 0);
  fs::path path = fs::current_path() / to_string(E) / (to_string(k) + EXT);
  File fi = File::openRead(path, true);
  string header = fi.readLine();

  u32 fileE, fileK, blockSize, nErrors, crc;
  u64 res64;

  u32 b1, nBits, start, nextK;
  vector<u32> check;
  if (sscanf(header.c_str(), HEADER_v10, &fileE, &fileK, &blockSize, &res64, &nErrors) == 5
      || sscanf(header.c_str(), HEADER_v11, &fileE, &fileK, &blockSize, &res64, &nErrors, &b1, &nBits, &start, &nextK, &crc) == 10) { 
    assert(E == fileE && k == fileK);
    if (!read(fi.get(), nWords(E), &check)) {
      log("Can't read PRP residue from '%s'\n", fi.name.c_str());
      throw "bad savefile";
    }
  } else {
    log("Can't parse PRP header '%s'\n", header.c_str());
    throw "bad savefile";
  }
  return {k, blockSize, res64, check, nErrors};
}

void PRPState::save(u32 E, const PRPState& state) {
  assert(state.check.size() == nWords(E));

  fs::path path = fs::current_path() / to_string(E) / (to_string(state.k) + EXT);
  File fo = File::openWrite(path);

  if (fprintf(fo.get(), HEADER_v10, E, state.k, state.blockSize, state.res64, state.nErrors) <= 0) {
    throw(ios_base::failure("can't write header"));
  }    
  write(fo.get(), state.check);
}

// --- P1 ---

void P1State::save(u32 E, u32 b1, u32 k, const P1State& state) {
  assert(state.data.size() == nWords(E));

  fs::path path = fs::current_path() / to_string(E) / (to_string(k) + EXT);
  File fo = File::openWrite(path);
  if (fprintf(fo.get(), HEADER_v2, E, b1, k, state.nextK, crc32(state.data)) <= 0) {
    throw(ios_base::failure("can't write header"));
  }    
  write(fo.get(), state.data);
}

P1State P1State::load(u32 E, u32 b1, u32 k) {
  fs::path path = fs::current_path() / to_string(E) / (to_string(k) + EXT);
  File fi = File::openRead(path, true);
  string header = fi.readLine();
  u32 fileE, fileB1, fileK, nextK, crc;
  vector<u32> data;
  if (sscanf(header.c_str(), HEADER_v2, &fileE, &fileB1, &fileK, &nextK, &crc) == 5) {
    assert(fileE == E && fileB1 == b1 && fileK == k);
    if (!read(fi.get(), nWords(E), &data)) {
      log("Can't read P-1 residue from '%s'\n", fi.name.c_str());
      throw "bad savefile";
    }
    
    if (crc != crc32(data)) {
      log("CRC in '%s' : found %u expected %u\n", fi.name.c_str(), crc, crc32(data));
      throw "bad savefile";
    }

    return {nextK, data};
  } else {
    log("Can't parse P-1 header '%s'\n", header.c_str());
    throw "bad savefile";
  }
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

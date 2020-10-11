// GpuOwl Mersenne primality tester; Copyright (C) Mihai Preda.

#include "checkpoint.h"
#include "File.h"
#include "Blake2.h"
#include "Args.h"

#include <filesystem>
#include <functional>
#include <ios>
#include <cassert>
#include <cinttypes>


namespace fs = std::filesystem;

namespace {

u32 nWords(u32 E) { return (E - 1) / 32 + 1; }

error_code& noThrow() {
  static error_code dummy;
  return dummy;
}

}

vector<u32> Saver::listIterations(const string& prefix, const string& ext) {
  vector<u32> ret;
  if (!fs::exists(base)) { fs::create_directory(base); }
  for (auto entry : fs::directory_iterator(base)) {
    if (entry.is_regular_file()) {
      string name = entry.path().filename().string();
      u32 dot = name.find('.');
      if (name.size() >= prefix.size() && name.substr(0, prefix.size()) == prefix
          && dot != string::npos && name.substr(dot) == ext) {
        assert(dot > prefix.size());
        size_t end = 0;
        u32 k = std::stoul(name.substr(prefix.size(), dot), &end);
        if (end != dot - prefix.size()) {
          log("Savefile ignored: '%s'\n", name.c_str());
        } else {
          ret.push_back(k);
        }
      }
    }
  }
  return ret;
}

vector<u32> Saver::listIterations() {
  return listIterations(to_string(E) + '-', ".prp");  
}

void Saver::cleanup(u32 E, const Args& args) {
  fs::path here = fs::current_path();
  fs::path trash = args.masterDir.empty() ? here / "trashbin" : (args.masterDir / "trashbin");
  fs::remove_all(trash, noThrow());
  fs::create_directory(trash, noThrow());
  fs::rename(here / to_string(E), trash / to_string(E), noThrow());
  
  // fs::remove_all(base, noThrow);  // redundant
}

float Saver::value(u32 k) {
  assert(k > 0);
  u32 dist = (k < E) ? (E - k) : 1;
  u32 nice = 1;

  while (k % 10 == 0) {
    k /= 10;
    nice *= 10;
  }

  if (k % 5 == 0) {
    k /= 5;
    nice *= 5;
  }
  
  while (k % 2 == 0) {
    k /= 2;
    nice *= 2;
  }
  
  return nice / float(dist);
}

Saver::Saver(u32 E, u32 nKeep, u32 b1, u32 startFrom) : E{E}, nKeep{max(nKeep, 5u)}, b1{b1} {
  scan(startFrom);
}

void Saver::scan(u32 upToK) {
  lastK = 0;
  minValPRP = {};
  
  vector<u32> iterations = listIterations();
  for (u32 k : iterations) {
    if (k <= upToK) {
      minValPRP.push({value(k), k});
      lastK = max(lastK, k);
    }
  }
}

void Saver::deleteBadSavefiles(u32 kBad, u32 currentK) {
  assert(kBad <= currentK);
  vector<u32> iterations = listIterations();
  for (u32 k : iterations) {
    if (k >= kBad && k <= currentK) {
      log("Deleting bad savefile @ %u\n", k);
      del(k);
    }
  }
  scan(kBad);
}

void Saver::del(u32 k) {
  // log("Note: deleting savefile %u\n", k);
  fs::remove(pathPRP(k), noThrow()); 
  fs::remove(pathP1(k), noThrow());
}

void Saver::savedPRP(u32 k) {
  assert(k >= lastK);
  lastK = k;
  while (minValPRP.size() >= nKeep) {
    auto kDel = minValPRP.top().second;
    minValPRP.pop();
    del(kDel);
  }
  minValPRP.push({value(k), k});
}

namespace {

vector<u32> makeVect(u32 size, u32 elem0) {
  vector<u32> v(size);
  v[0] = elem0;
  return v;
}

}

// --- PRP ---

PRPState Saver::loadPRP(u32 iniBlockSize) {
  if (lastK == 0) {
    log("PRP starting from beginning\n");
    u32 blockSize = iniBlockSize ? iniBlockSize : 500;
    return {0, blockSize, 3, makeVect(nWords(E), 1), 0};
  } else {
    return loadPRPAux(lastK);
  }
}

PRPState Saver::loadPRPAux(u32 k) {
  assert(k > 0);
  fs::path path = pathPRP(k);
  File fi = File::openRead(path, true);
  string header = fi.readLine();

  u32 fileE, fileK, blockSize, nErrors, crc;
  u64 res64;
  vector<u32> check;
  u32 b1, nBits, start, nextK;
  if (sscanf(header.c_str(), PRP_v12, &fileE, &fileK, &blockSize, &res64, &nErrors, &crc) == 6) {
    assert(E == fileE && k == fileK);
    check = fi.readWithCRC<u32>(nWords(E), crc);
  } else if (sscanf(header.c_str(), PRP_v10, &fileE, &fileK, &blockSize, &res64, &nErrors) == 5
             || sscanf(header.c_str(), PRP_v11, &fileE, &fileK, &blockSize, &res64, &nErrors, &b1, &nBits, &start, &nextK, &crc) == 10) { 
    assert(E == fileE && k == fileK);
    check = fi.read<u32>(nWords(E));
  } else {
    log("In file '%s': bad header '%s'\n", fi.name.c_str(), header.c_str());
    throw "bad savefile";
  }
  return {k, blockSize, res64, check, nErrors};
}

void Saver::savePRP(const PRPState& state) {
  assert(state.check.size() == nWords(E));
  u32 k = state.k;
  
  fs::path path = pathPRP(k);
  {
    File fo = File::openWrite(path);

    if (fo.printf(PRP_v12, E, k, state.blockSize, state.res64, state.nErrors, crc32(state.check)) <= 0) {
      throw(ios_base::failure("can't write header"));
    }    
    fo.write(state.check);
  }
  loadPRPAux(k);
  savedPRP(k);
}

// --- P1 ---

P1State Saver::loadP1(u32 k) {
  fs::path path = pathP1(k);
  File fi = File::openRead(path, true);
  string header = fi.readLine();
  u32 fileE, fileB1, fileK, nextK, crc;
  if (sscanf(header.c_str(), P1_v2, &fileE, &fileB1, &fileK, &nextK, &crc) != 5) {
    log("In file '%s': bad header '%s'\n", fi.name.c_str(), header.c_str());
    throw "bad savefile";
  }
  
  assert(fileE == E && fileB1 == b1 && fileK == k);
  return {nextK, fi.readWithCRC<u32>(nWords(E), crc)};
}

void Saver::saveP1(u32 k, const P1State& state) {
  assert(state.data.size() == nWords(E));
  assert(state.nextK == 0 || state.nextK >= k);

  {
    File fo = File::openWrite(pathP1(k));
    if (fo.printf(P1_v2, E, b1, k, state.nextK, crc32(state.data)) <= 0) {
      throw(ios_base::failure("can't write header"));
    }
    fo.write(state.data);
  }

  loadP1(k);
}

// --- P1Final ---

vector<u32> Saver::loadP1Final() {
  fs::path path = pathP1Final();
  File fi = File::openRead(path, true);
  string header = fi.readLine();
  u32 fileE, fileB1, crc;
  if (sscanf(header.c_str(), P1Final_v1, &fileE, &fileB1, &crc) != 3) {
    log("In file '%s': bad header '%s'\n", fi.name.c_str(), header.c_str());
    throw "bad savefile";
  }
  
  assert(fileE == E && fileB1 == b1);
  return fi.readWithCRC<u32>(nWords(E), crc);
}

void Saver::saveP1Final(const vector<u32>& data) {
  assert(data.size() == nWords(E));
  {
    File fo = File::openWrite(pathP1Final());
    if (fo.printf(P1Final_v1, E, b1, crc32(data)) <= 0) {
      throw(ios_base::failure("can't write header"));
    }
    fo.write(data);
  }

  loadP1Final();
}

// --- P2 ---

u32 Saver::loadP2() {
  fs::path path = pathP2();
  File fi = File::openRead(path);
  if (!fi) {
    return 0;
  } else {    
    string header = fi.readLine();    
    u32 fileE, fileB1, b2;    
    if (sscanf(header.c_str(), P2_v2, &fileE, &fileB1, &b2) != 3) {
      log("In file '%s' wrong header '%s'\n", fi.name.c_str(), header.c_str());
      throw "bad savefile";
    }
    assert(fileE == E && fileB1 == b1);
    return b2;
  }
}

void Saver::saveP2(u32 b2) {
  File fo = File::openWrite(pathP2());
  if (fo.printf(P2_v2, E, b1, b2) <= 0) {
    throw(ios_base::failure("can't write header"));
  }
}

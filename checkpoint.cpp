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
  // std::sort(ret.begin(), ret.end());
  return ret;
}

void Saver::cleanup(u32 E) {
  fs::path here = fs::current_path();
  
  fs::path trash = here / "trashbin";
  fs::remove_all(trash, noThrow());
  fs::create_directory(trash, noThrow());
  fs::rename(here / to_string(E), trash / to_string(E), noThrow());
  
  // fs::remove_all(base, noThrow);  // redundant
}

float Saver::value(u32 k) {
  assert(k > 0);
  u32 dist = (k < E) ? (E - k) : 1;
  return (1u << __builtin_ctz(k)) / float(dist);
}

Saver::Saver(u32 E, u32 nKeep, vector<u32> b1s) : E{E}, nKeep{max(nKeep, 4u)}, b1s{b1s} {
  std::sort(b1s.begin(), b1s.end());
  
  vector<u32> iterations = listIterations(to_string(E) + '-', ".prp");
  for (u32 k : iterations) {
    minValPRP.push({value(k), k});
    lastK = max(lastK, k);
  }

  /*
  if (!b1s.empty()) {
    iterations = listIterations(to_string(E) + '-' + to_string(b1s.back()) + '-', ".p2");
    for (u32 k : iterations) {
      minValP2.push({value(k), k});
      lastB2 = max(lastK, k);
    }
  }
  */
}

void Saver::del(u32 k) {
  // log("Note: deleting savefile %u\n", k);
  fs::remove(pathPRP(k), noThrow());
  for (u32 b1 : b1s) { fs::remove(pathP1(b1, k), noThrow()); }
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

/*
void Saver::savedP2(u32 b2) {
  assert(!b1s.empty());
  
  assert(b2 >= lastB2);
  lastB2 = b2;
  while (minValP2.size() >= nKeep) {
    auto kDel = minValP2.top().second;
    minValP2.pop();
    fs::remove(pathP2(b1s.back(), kDel), noThrow());
  }
  minValP2.push({value(b2), b2});
}
*/

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
    log("PRP starting from the beginning\n");
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

void Saver::save(const PRPState& state) {
  assert(state.check.size() == nWords(E));
  u32 k = state.k;
  
  fs::path path = pathPRP(k);
  {
    File fo = File::openWrite(path);

    if (fprintf(fo.get(), PRP_v12, E, k, state.blockSize, state.res64, state.nErrors, crc32(state.check)) <= 0) {
      throw(ios_base::failure("can't write header"));
    }    
    fo.write(state.check);
  }
  loadPRPAux(k);
  savedPRP(k);
}

// --- P1 ---

P1State Saver::loadP1(u32 b1, u32 k) {
  fs::path path = pathP1(b1, k);
  File fi = File::openRead(path, true);
  string header = fi.readLine();
  u32 fileE, fileB1, fileK, nextK, crc;
  if (sscanf(header.c_str(), P1_v2, &fileE, &fileB1, &fileK, &nextK, &crc) != 5) {
    log("In file '%s': bad header '%s'\n", fi.name.c_str(), header.c_str());
    throw "bad savefile";
  }
  
  assert(fileE == E && fileB1 == b1 && fileK == k);
  return {k, fi.readWithCRC<u32>(nWords(E), crc)};
}

void Saver::save(u32 b1, u32 k, const P1State& state) {
  assert(state.data.size() == nWords(E));

  {
    File fo = File::openWrite(pathP1(b1, k));
    if (fprintf(fo.get(), P1_v2, E, b1, k, state.nextK, crc32(state.data)) <= 0) {
      throw(ios_base::failure("can't write header"));
    }
    fo.write(state.data);
  }

  loadP1(b1, k);
}

// --- P2 ---

u32 Saver::loadP2(u32 b1) {
  fs::path path = pathP2(b1);
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

void Saver::saveP2(u32 b1, u32 b2) {
  File fo = File::openWrite(pathP2(b1));
  if (fprintf(fo.get(), P2_v2, E, b1, b2) <= 0) {
    throw(ios_base::failure("can't write header"));
  }
}

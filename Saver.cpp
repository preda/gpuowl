// GpuOwl Mersenne primality tester; Copyright (C) Mihai Preda.

#include "Saver.h"
#include "File.h"
#include "Blake2.h"
#include "Args.h"
#include "common.h"

#include <filesystem>
#include <functional>
#include <ios>
#include <cassert>
#include <cinttypes>
#include <string>

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
  if (args.clean) {
    fs::path here = fs::current_path();
    fs::remove_all(here / to_string(E), noThrow());
  }
}

float Saver::value(u32 k) {
  assert(k > 0);
  u32 dist = (k < E) ? (E - k) : 1;
  u32 nice = 1;

  while (k % 2 == 0) {
    k /= 2;
    nice *= 2;

    if (k % 5 == 0) {
      k /= 5;
      nice *= 5;
    }
  }
      
  return nice / float(dist);
}

Saver::Saver(u32 E, u32 nKeep, u32 startFrom) : E{E}, nKeep{max(nKeep, 5u)} {
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
}

void Saver::savedPRP(u32 k) {
  assert(k >= lastK);
  while (minValPRP.size() >= nKeep) {
    auto kDel = minValPRP.top().second;
    minValPRP.pop();
    del(kDel);
  }
  lastK = k;
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
    u32 blockSize = iniBlockSize ? iniBlockSize : 400;
    return {0, blockSize, 3, makeVect(nWords(E), 1), 0};
  } else {
    return loadPRPAux(lastK);
  }
}

PRPState Saver::loadPRPAux(u32 k) {
  assert(k > 0);
  fs::path path = pathPRP(k);
  File fi = File::openReadThrow(path);
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

P1State Saver::loadP1() {
  try {
    File fi = File::openReadThrow(pathP1());

    string header = fi.readLine();
    u32 fileE, fileB1, fileK, fileBlock;
    if (sscanf(header.c_str(), P1_v3, &fileE, &fileB1, &fileK, &fileBlock) != 4) {
      log("In file '%s': bad header '%s'\n", fi.name.c_str(), header.c_str());
      throw "bad savefile";
    }

    assert(fileE == E);

    auto data  = fi.readChecked<u32>(nWords(E));
    return {fileB1, fileK, fileBlock, data};

  } catch (const fs::filesystem_error& e) {
    log("P1: no savefile found, starting from the beginning\n");
    return P1State{}; // {0, 0, 0, {}};
  }
}

void Saver::saveP1(const P1State& state) {
  assert(state.data.size() == nWords(E));
  assert(state.B1);
  {
    File fo = File::openWrite(pathP1() + ".new");
    if (fo.printf(P1_v3, E, state.B1, state.k, state.blockSize) <= 0) {
      throw(ios_base::failure("can't write header"));
    }
    fo.writeChecked(state.data);
  }
  cycle(pathP1());
}

void Saver::cycle(const fs::path& name) {
  fs::remove(name + ".bak");
  fs::rename(name, name + ".bak", noThrow());
  fs::rename(name + ".new", name, noThrow());
}

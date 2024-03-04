// GpuOwl Mersenne primality tester; Copyright (C) Mihai Preda.

#include "Saver.h"
#include "File.h"
#include "Args.h"
#include "common.h"

#include <filesystem>
#include <ios>
#include <cassert>

namespace fs = std::filesystem;

namespace {

u32 nWords(u32 E) { return (E - 1) / 32 + 1; }

error_code& noThrow() {
  static error_code dummy;
  return dummy;
}

}

vector<u32> Saver::listIterations(const string& ext) {
  const string prefix = to_string(E) + '-';
  vector<u32> ret;
  if (!fs::exists(base)) { fs::create_directory(base); }
  for (const auto& entry : fs::directory_iterator(base)) {
    if (entry.is_regular_file()) {
      string name = entry.path().filename().string();
      auto dot = name.find('.');
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

Saver::Saver(u32 E)
  : E{E} {
  scan();
}

void Saver::scan(u32 upToK) {
  lastK = 0;
  minValPRP = {};
  
  vector<u32> iterations = listIterationsPRP();
  for (u32 k : iterations) {
    if (k <= upToK) {
      minValPRP.push({value(k), k});
      lastK = max(lastK, k);
    }
  }
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

void Saver::cycle(const fs::path& name) {
  fs::remove(name + ".bak");
  fs::rename(name, name + ".bak", noThrow());
  fs::rename(name + ".new", name, noThrow());
}

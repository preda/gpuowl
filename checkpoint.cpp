// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "checkpoint.h"
#include "file.h"
#include <cassert>

std::string fileName(int E, const string &suffix) { return std::to_string(E) + suffix + ".owl"; }

static bool write(FILE *fo, const vector<u32> &v) {
  return fwrite(v.data(), v.size() * sizeof(u32), 1, fo);
}

static bool read(FILE *fi, u32 nWords, vector<u32> *v) {
  v->resize(nWords);
  return fread(v->data(), nWords * sizeof(u32), 1, fi);
}

struct PRPState_v5 {
  static constexpr const char *HEADER_R = R"(OWL 5
Comment: %255[^
]
Type: PRP
Exponent: %u
Iteration: %u
PRP-block-size: %u
Residue-64: 0x%016llx
Errors: %d
End-of-header:
\0)";

  static constexpr const char *SUFFIX = "";

  u32 k;
  u32 blockSize;
  int nErrors;
  u64 res64;
  vector<u32> check;
  
  bool load(u32 E) {
    u32 nWords = (E - 1)/32 + 1;
    u32 nBytes = (E - 1)/8 + 1;
    
    auto fi{openRead(fileName(E, SUFFIX))};
    if (!fi) { return false; }
    
    char buf[256];
    u32 fileE;
    if (!(fscanf(fi.get(), HEADER_R, buf, &fileE, &k, &blockSize, &res64, &nErrors) == 6)) { return false; }
    assert(E == fileE);
    check = vector<u32>(nWords);
    if (!fread(check.data(), nBytes, 1, fi.get())) { return false; }
    return true;
  }
};

bool PRPState::load_v5(u32 E) {
  PRPState_v5 v5;
  if (!v5.load(E)) { return false; }
  
  k = v5.k;
  blockSize = v5.blockSize;
  nErrors = v5.nErrors;
  res64 = v5.res64;
  check = move(v5.check);
  return true;
}

void PRPState::loadInt(u32 E, u32 iniBlockSize) {    
  u32 nWords = (E - 1) / 32 + 1;
  if (auto fi{openRead(fileName(E, SUFFIX))}) {
    char line[256];
    u32 fileE;    
    bool ok = fgets(line, sizeof(line), fi.get())
      && sscanf(line, HEADER, &fileE, &k, &blockSize, &res64, &nErrors) == 5
      && read(fi.get(), nWords, &check);
    if (!ok) {
      log("Invalid PRP checkpoint for exponent %u\n", E);
      throw "load";
    }
    assert(E == fileE);
  } else if (!load_v5(E)) {
    k = 0;
    blockSize = iniBlockSize;
    nErrors = 0;
    res64 = 3;
    check = vector<u32>(nWords);
    check[0] = 1;
  }
}

bool PRPState::saveImpl(u32 E, const string &name) {
  assert(check.size() == (E - 1) / 32 + 1);
  auto fo(openWrite(name));
  return fo
    && fprintf(fo.get(),  HEADER, E, k, blockSize, res64, nErrors) > 0
    && write(fo.get(), check);
}

string PRPState::durableName() {
  return k && (k % 20'000'000 == 0) ? "."s + to_string(k/1'000'000)+"M" : ""s;
}

void PFState::loadInt(u32 E, u32 iniB1) {
  u32 nWords = (E - 1)/32 + 1;    
  if (auto fi{openRead(fileName(E, SUFFIX))}) {
    char line[256];
    u32 fileE;
    if (!fgets(line, sizeof(line), fi.get())
        || !(sscanf(line, HEADER, &fileE, &k, &kEnd, &B1) == 4)
        || !read(fi.get(), nWords, &base)) {
      log("Could not load P-1 savefile for %u\n", E);
      throw "load";
    }
    assert(E == fileE);
  } else {
    k = 0;
    kEnd = 0; // not known yet.
    B1 = iniB1;
    base = vector<u32>(nWords);
    base[0] = 1;
  }
}  

bool PFState::saveImpl(u32 E, const string &name) {
  assert(base.size() == (E - 1) / 32 + 1);
    
  auto fo(openWrite(name));
  return fo
    && fprintf(fo.get(), HEADER, E, k, kEnd, B1) > 0
    && write(fo.get(), base);
}

void PRPFState::loadInt(u32 E, u32 iniB1, u32 iniBlockSize) {
  u32 nWords = (E - 1) / 32 + 1;        
  if (auto fi{openRead(fileName(E, SUFFIX))}) {
    char line[256];
    u32 fileE;
    if (fgets(line, sizeof(line), fi.get())
        && sscanf(line, HEADER, &fileE, &k, &B1, &blockSize, &res64) == 5
        && read(fi.get(), nWords, &base)
        && read(fi.get(), nWords, &check)) {
      assert(E == fileE);
    } else {
      throw "PRPF load";
    }
  } else {
    PFState pf = PFState::load(E, iniB1);
    if (pf.isCompleted()) {
      k = 0;
      B1 = pf.B1;
      blockSize = iniBlockSize;
      base = move(pf.base);
      assert(base.size() == nWords);
      res64 = (u64(base[1]) << 32) | base[0];
      check = vector<u32>(nWords);
      check[0] = 1;
    } else {
      throw "PF not completed";
    }
  }
}

bool PRPFState::saveImpl(u32 E, string name) {
  u32 nWords = (E - 1) / 32 + 1;
  assert(base.size() == nWords && check.size() == nWords);
  
  auto fo(openWrite(name));
  return fo
    && fprintf(fo.get(), HEADER, E, k, B1, blockSize, res64) == 5
    && write(fo.get(), base)
    && write(fo.get(), check);    
}

bool PRPFState::canProceed(u32 E, u32 B1) {
  return (B1 == 0) || openRead(fileName(E, SUFFIX)) || PFState::load(E, B1).isCompleted();
}

void TFState::loadInt(u32 E) {    
  if (auto fi{openRead(fileName(E, SUFFIX))}) {
    u32 fileE;
    if (fscanf(fi.get(), HEADER, &fileE, &bitLo, &bitHi, &nDone, &nTotal) == 5) {
      assert(E == fileE);
      assert(bitLo < bitHi);
    } else {
      log("Can't parse file '%s'\n", fileName(E, SUFFIX).c_str());
      throw "parse";
    }
  } else {
    bitLo = bitHi = nDone = nTotal = 0;
  }
}

bool TFState::saveImpl(u32 E, string name) {
  auto fo(openWrite(name));
  return fo && fprintf(fo.get(), HEADER, E, bitLo, bitHi, nDone, nTotal) > 0;
}

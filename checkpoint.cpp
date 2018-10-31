// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "checkpoint.h"
#include "file.h"

#include <cassert>
#include <cmath>
#include <gmp.h>

// Residue from compacted words.
u64 residue(const vector<u32> &words) { return (u64(words[1]) << 32) | words[0]; }

static std::string fileName(int E, const string &suffix) { return std::to_string(E) + suffix + ".owl"; }

void PRPState::save(u32 E) {
  string tempFile = fileName(E, "-temp"s + SUFFIX);
  if (!saveImpl(E, tempFile)) {
    throw "can't save";
  }
  
  string prevFile = fileName(E, "-prev"s + SUFFIX);
  remove(prevFile.c_str());
  
  string saveFile = fileName(E, SUFFIX);
  rename(saveFile.c_str(), prevFile.c_str());
  rename(tempFile.c_str(), saveFile.c_str());
  
  string persist = durableName();
  if (!persist.empty() && !saveImpl(E, fileName(E, persist + SUFFIX))) {
    throw "can't save";
  }
}

static bool write(FILE *fo, const vector<u32> &v) {
  return fwrite(v.data(), v.size() * sizeof(u32), 1, fo);
}

static bool read(FILE *fi, u32 nWords, vector<u32> *v) {
  v->resize(nWords);
  return fread(v->data(), nWords * sizeof(u32), 1, fi);
}

static void powerSmooth(mpz_t a, u32 exp, u32 B1, u32 B2 = 0) {
  if (B2 == 0) { B2 = B1; }
  assert(B2 >= sqrt(B1));

  mpz_set_ui(a, exp);
  mpz_mul_2exp(a, a, 20); // boost 2s.

  mpz_t b; mpz_init(b);
  
  for (int k = log2(B1); k > 1; --k) {
    u32 limit = pow(B1, 1.0 / k);
    mpz_primorial_ui(b, limit);
    mpz_mul(a, a, b);
  }
  
  mpz_primorial_ui(b, B2);
  mpz_mul(a, a, b);
  mpz_clear(b);
}

// "Rev" means: most significant bit first (at index 0).
static vector<bool> powerSmoothBitsRev(u32 exp, u32 B1) {
  mpz_t a;
  mpz_init(a);
  powerSmooth(a, exp, B1);
  int nBits = mpz_sizeinbase(a, 2);
  vector<bool> bits;
  for (int i = nBits - 1; i >= 0; --i) { bits.push_back(mpz_tstbit(a, i)); }
  assert(int(bits.size()) == nBits);
  mpz_clear(a);
  return bits;
}

static vector<u32> makeVect(u32 size, u32 elem0) {
  vector<u32> v(size);
  v[0] = elem0;
  return v;
}

PRPState PRPState::initStage1(u32 iniB1, u32 iniBlockSize, const vector<u32> &iniBase) {
  stage = 1;
  k = 0;
  B1 = iniB1;
  blockSize = iniBlockSize;
  base = iniBase;
  res64  = residue(base);
  u32 nWords = iniBase.size(); // (E - 1) / 32 + 1;
  check = gcdAcc = makeVect(nWords, 1);
  return *this;
}

void PRPState::loadInt(u32 E, u32 wantB1, u32 iniBlockSize) {
  u32 nWords = (E - 1) / 32 + 1;
  string name = fileName(E, SUFFIX);  
  auto fi{openRead(name)};
  if (!fi) {
    log("%s not found, starting from the beginnig.\n", name.c_str());
    k = 0;
    B1 = wantB1;
    blockSize = iniBlockSize;

    if (B1 > 0) {
      stage = 0;
      res64 = 1;
      base = makeVect(nWords, 1);
      base[0] = 1;
      basePower = powerSmoothBitsRev(E, B1);
      log("powerSmooth(%u, %u) has %u bits\n", E, B1, u32(basePower.size()));
    } else {
      initStage1(B1, blockSize, makeVect(nWords, 3));
    }
    return;
  }
  
  char line[256];
  if (!fgets(line, sizeof(line), fi.get())) {
    log("Invalid savefile '%s'\n", name.c_str());
    throw("invalid savefile");
  }
  
  stage = 1;
  u32 fileE = 0;
  u32 nBaseBits = 0;

  if (sscanf(line, HEADER_v7, &fileE, &k, &B1, &blockSize, &res64) == 5) {
    assert(E == fileE);
    if (B1 != wantB1) { log("B1 mismatch: using B1=%u from '%s' instead of %u\n", B1, name.c_str(), wantB1); }
    if (!read(fi.get(), nWords, &check)) { throw("load: error read check"); }
    assert(stage == 1);
    if (B1 == 0) {
      base = makeVect(nWords, 3);
      gcdAcc = makeVect(nWords, 1);
    } else {
      bool ok = read(fi.get(), nWords, &base);
      assert(ok);
      gcdAcc = makeVect(nWords, 1);
    }        
  } else if (sscanf(line, HEADER_v8, &fileE, &k, &B1, &blockSize, &res64, &stage, &nBaseBits) == 7) {
    assert(E == fileE);
    if (B1 != wantB1) { log("B1 mismatch: using B1=%u from '%s' instead of %u\n", B1, name.c_str(), wantB1); }
    if (!read(fi.get(), nWords, &check)) { throw("load: error read check"); }
    
    if (stage == 0) {
      std::swap(check, base);
      assert(res64 == residue(base));
      assert(B1 != 0);
      assert(k > 0 && k < nBaseBits);
      basePower = powerSmoothBitsRev(E, B1);
      assert(nBaseBits == basePower.size());
    } else {
      assert(stage == 1);
      if (B1 == 0) {
        base = makeVect(nWords, 3);
        gcdAcc = makeVect(nWords, 1);
      } else {
        bool ok = read(fi.get(), nWords, &base) && read(fi.get(), nWords, &gcdAcc);
        assert(ok);
      }          
    }
  } else {
    log("Invalid savefile '%s'\n", name.c_str());
    throw("invalid savefile");    
  }

  log("%s loaded: k %u, B1 %u, block %u, res64 %016llx, stage %u, baseBits %u\n",
      name.c_str(), k, B1, blockSize, res64, stage, nBaseBits);
}

bool PRPState::saveImpl(u32 E, const string &name) {
  u32 nWords = (E - 1) / 32 + 1;
  assert(check.size() == nWords);

  auto fo(openWrite(name));
  return
    fo
    && fprintf(fo.get(), HEADER_v8, E, k, B1, blockSize, res64, stage, u32(basePower.size())) > 0
    && write(fo.get(), check)
    && (B1 == 0 || stage == 0 || (write(fo.get(), base) && write(fo.get(), gcdAcc)));
}

string PRPState::durableName() {
  if (k == 0 && B1 != 0) { return ".0"; }
  if (k && (k % 20'000'000 == 0)) { return "."s + to_string(k/1'000'000)+"M"; }
  return "";
}

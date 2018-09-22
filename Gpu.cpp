// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "Gpu.h"

#include "checkpoint.h"
#include "stats.h"
#include "timeutil.h"
#include "ghzdays.h"
#include "args.h"
#include "Kset.h"

#include <gmp.h>
#include <cmath>
#include <signal.h>

// Check that mpz "_ui" takes 64-bit.
static_assert(sizeof(long) == 8, "size long");

std::string makeLogStr(int E, int k, u64 res, const StatsInfo &info, u32 nIters = 0) {
  int end = nIters ? nIters : (((E - 1) / 1000 + 1) * 1000);
  float percent = 100 / float(end);
  
  int etaMins = (end - k) * info.mean * (1 / 60000.f) + .5f;
  int days  = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins  = etaMins % 60;

  char buf[256];
  string ghzStr;
  /*
  if (ghzMsPerIt) {
    snprintf(buf, sizeof(buf), " (%.1f GHz-days/day)", ghzMsPerIt / info.mean);
    ghzStr = buf;
  }
  */
  
  snprintf(buf, sizeof(buf), "%8d/%d [%5.2f%%], %.2f ms/it [%.2f, %.2f]%s; ETA %dd %02d:%02d; %016llx",
           k, nIters ? nIters : E, k * percent, info.mean, info.low, info.high, ghzStr.c_str(), days, hours, mins, res);
  return buf;
}

void doLog(int E, int k, long timeCheck, u64 res, bool checkOK, Stats &stats) {
  // std::string errors = !nErrors ? "" : ("; (" + std::to_string(nErrors) + " errors)");
  log("%s %s (check %.2fs)\n",
      checkOK ? "OK" : "EE",
      makeLogStr(E, k, res, stats.getStats()).c_str(),
      timeCheck * .001f);
  stats.reset();
}

void doSmallLog(int E, int k, u64 res, Stats &stats) {
  log("   %s\n", makeLogStr(E, k, res, stats.getStats()).c_str());
  stats.reset();
}

static void powerSmooth(mpz_t a, u32 exp, u32 B1, u32 B2 = 0) {
  if (B2 == 0) { B2 = B1; }
  assert(B2 >= sqrt(B1));

  mpz_set_ui(a, u64(exp) << 20); // boost 2s.

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

// "Rev" means: most significant bit first.
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

static string GCD(u32 exp, const vector<u32> &bits, u32 sub) {
  mpz_t b;
  mpz_init(b);
  mpz_import(b, bits.size(), -1 /*order: LSWord first*/, sizeof(u32), 0 /*endianess: native*/, 0 /*nails*/, bits.data());
  mpz_sub_ui(b, b, sub);
  assert(mpz_sizeinbase(b, 2) <= exp);
  assert(mpz_cmp_ui(b, 0)); // b != 0.
  
  mpz_t m;
  // m := 2^exp - 1.
  mpz_init_set_ui(m, 1);
  mpz_mul_2exp(m, m, exp);
  mpz_sub_ui(m, m, 1);
  assert(mpz_sizeinbase(m, 2) == exp);
    
  mpz_gcd(m, m, b);
    
  mpz_clear(b);

  if (mpz_cmp_ui(m, 1) == 0) { return ""; }

  char *buf = mpz_get_str(nullptr, 10, m);
  string ret = buf;
  free(buf);

  mpz_clear(m);
  return ret;
}

// static u32 getB1(u32 exp, float prpTimeFraction) { return exp * prpTimeFraction / 1.4429f; }

static volatile int stopRequested = 0;
void (*oldHandler)(int) = 0;
void myHandler(int dummy) { stopRequested = 1; }

Gpu::~Gpu() {}

string Gpu::factorPM1(u32 E, const Args &args) {
  u32 B1 = args.getB1();
  PFState loaded = PFState::load(E, B1);

  u32 N = this->getFFTSize();
  u32 k  = loaded.k;

  vector<bool> bits = powerSmoothBitsRev(E, B1);
  assert(bits.front()); // most significant bit is 1.
  
  this->writeData(move(loaded.base));
  
  log("P-1 M(%d), FFT %dK, %.2f bits/word, B1 %u, at %u, %016llx\n", E, N/1024, E / float(N), B1, k, this->dataResidue());

  // oldHandler = signal(SIGINT, myHandler);
  
  Stats stats;  
  Timer timer;

  const u32 step = 1000;
  while (k < bits.size()) {
    assert(k % step == 0);
    u32 nIts = min(u32(bits.size() - k), 1000u);    
    vector<bool> muls(bits.begin() + k, bits.begin() + (k + nIts));
    this->dataLoop(muls);
    this->finish();
    stats.add(timer.deltaMillis() / float(nIts));
    k += nIts;
    
    if (k % 10000 == 0) {
      log("   %s\n", makeLogStr(E, k, this->dataResidue(), stats.getStats(), bits.size()).c_str());
      stats.reset();
      PFState{k, u32(bits.size()), B1, this->readData()}.save(E);
    }
  }

  assert(k == bits.size());
  if (k != loaded.k) { PFState{k, u32(bits.size()), B1, this->readData()}.save(E); }

  return GCD(E, this->readData(), 1);
}

// Residue from compacted words.
static u64 residue(const std::vector<u32> &words) { return (u64(words[1]) << 32) | words[0]; }

vector<u32> bitNeg(const vector<u32> &v) {
  vector<u32> ret;
  ret.reserve(v.size());
  for (auto x : v) { ret.push_back(~x); }
  return ret;
}

static bool loadPRP(Gpu *gpu, u32 E, u32 B1, u32 desiredBlockSize, u32 *outK, u32 *outBlockSize, vector<u32> *outBase) {
  auto loaded = PRPState::load(E, B1, desiredBlockSize);    
  gpu->writeState(loaded.check, loaded.base, loaded.blockSize);

  u64 res64 = gpu->dataResidue();
  bool ok = (res64 == loaded.res64);
  gpu->updateCheck();
  // bool ok = resOk && gpu->checkAndUpdate(loaded.blockSize);
  log("%s loaded: %d/%d, B1 %u, blockSize %d, %016llx (expected %016llx)\n",
      ok ? "OK" : "EE",  loaded.k, E, B1, loaded.blockSize, res64, loaded.res64);
  if (!ok) { return false; }
  
  *outK = loaded.k;
  *outBlockSize = loaded.blockSize;
  *outBase = loaded.base;
  return true;
}

bool Gpu::isPrimePRP(u32 E, const Args &args, u64 *outRes, u64 *outBaseRes, string *outFactor) {
  u32 B1 = args.getB1();

  u32 N = this->getFFTSize();
  log("PRP M(%d), FFT %dK, %.2f bits/word, B1 %u\n", E, N/1024, E / float(N), B1);
  
  u32 k = 0, blockSize = 0;
  vector<u32> base;

  if (!loadPRP(this, E, B1, args.blockSize, &k, &blockSize, &base)) { throw "error at start"; }
  
  const u64 baseRes64 = residue(base);
  *outBaseRes = baseRes64;
  const int checkStep = blockSize * blockSize;
  const u32 kEnd = E - 1; // Type-4 per http://www.mersenneforum.org/showpost.php?p=468378&postcount=209
  assert(k % blockSize == 0 && k < kEnd);
  
  oldHandler = signal(SIGINT, myHandler);

  int startK = k;
  Stats stats;

  // Number of sequential errors with no success in between. If this ever gets high enough, stop.
  int nSeqErrors = 0;

  bool isPrime = false;
  Timer timer;

  Kset kset(args.ksetFile);

  int nGcdAcc = 0;
  while (true) {
    assert(k % blockSize == 0);

    if (kEnd - k <= blockSize) {
      this->dataLoop(kEnd - k);
      auto words = this->roundtripData();
      u64 res64 = residue(words);
      isPrime = (words == base || words == bitNeg(base));

      log("%s %8d / %d, %016llx (base %016llx)\n", isPrime ? "PP" : "CC", kEnd, E, res64, baseRes64);
      
      *outRes = res64;
      *outBaseRes = baseRes64;
      int itersLeft = blockSize - (kEnd - k);
      assert(itersLeft > 0);
      this->dataLoop(itersLeft);
      k += blockSize;
    } else {
      do {
        u32 kToTest = kset.getFirstAfter(k);
        assert(kToTest > k);
        u32 nIters = min(blockSize - (k % blockSize), kToTest - k);
        assert(nIters > 0);
        this->dataLoop(nIters);
        k += nIters;
        if (k == kToTest) {
          bool isFirst = nGcdAcc == 0;
          this->gcdAccumulate(isFirst);
          ++nGcdAcc;
        }
      } while (k % blockSize != 0);
    }

    u64 res64 = this->dataResidue();
    // this->finish();
    auto delta = timer.deltaMillis();
    stats.add(delta * (1.0f / blockSize));
    bool doStop = stopRequested;
    if (doStop) {
      log("Stopping, please wait..\n");
      signal(SIGINT, oldHandler);
    }

    bool doCheck = (k % checkStep == 0) || (k >= kEnd) || doStop || (k - startK == 2 * blockSize);

    if (!doCheck) {
      this->updateCheck();
      if (k % 10000 == 0) {
        doSmallLog(E, k, res64, stats);
        if (args.timeKernels) { this->logTimeKernels(); }
      }
      continue;
    }

    vector<u32> check = this->roundtripCheck();
    vector<u32> acc = this->readAcc();
    this->startCheck(blockSize);
    string factor;
    if (nGcdAcc > 0) {
      Timer gcdTimer;
      factor = GCD(E, acc, 0);
      log("GCD: %d MULs in %.1fs, '%s'\n", nGcdAcc, gcdTimer.deltaMillis()/1000.0f, factor.c_str());
      nGcdAcc = 0;
    }
    bool ok = this->finishCheck();

    if (!factor.empty()) {
      *outRes = 0;
      *outFactor = factor;
      return false;
    }
    
    bool doSave = k < kEnd && ok;
    if (doSave) { PRPState{k, blockSize, res64, check, base}.save(E, B1); }
    
    doLog(E, k, timer.deltaMillis(), res64, ok, stats);
    
    if (ok) {
      if (k >= kEnd) { return isPrime; }
      nSeqErrors = 0;
    } else {
      if (++nSeqErrors > 2) {
        log("%d sequential errors, will stop.\n", nSeqErrors);
        throw "too many errors";
      }
      
      // re-try failed load once.
      if (!loadPRP(this, E, B1, blockSize, &k, &blockSize, &base) &&
          !loadPRP(this, E, B1, blockSize, &k, &blockSize, &base)) {
        throw "errors on retry";
      }
    }
    if (args.timeKernels) { this->logTimeKernels(); }
    if (doStop) { throw "stop requested"; }
  }
}

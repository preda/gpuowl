// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "Gpu.h"

#include "checkpoint.h"
#include "stats.h"
#include "timeutil.h"
#include "args.h"
#include "GCD.h"

#include <gmp.h>
#include <cmath>
#include <signal.h>

static_assert(sizeof(long) == 8, "size long");

Gpu::Gpu() :
  gcd(make_unique<GCD>())
{
}

std::string makeLogStr(int E, int k, u64 res, const StatsInfo &info, u32 nIters = 0) {
  int end = nIters ? nIters : (((E - 1) / 1000 + 1) * 1000);
  float percent = 100 / float(end);
  
  int etaMins = (end - k) * info.mean * (1 / 60000.f) + .5f;
  int days  = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins  = etaMins % 60;

  char buf[256];
  string ghzStr;
  
  snprintf(buf, sizeof(buf), "%8d/%d [%5.2f%%], %.2f ms/it [%.2f, %.2f]%s; ETA %dd %02d:%02d; %016llx",
           k, nIters ? nIters : E, k * percent, info.mean, info.low, info.high, ghzStr.c_str(), days, hours, mins, res);
  return buf;
}

void doLog(int E, int k, long timeCheck, u64 res, bool checkOK, Stats &stats) {
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

static volatile int stopRequested = 0;
void (*oldHandler)(int) = 0;
void myHandler(int dummy) { stopRequested = 1; }

Gpu::~Gpu() {}

// Residue from compacted words.
static u64 residue(const std::vector<u32> &words) { return (u64(words[1]) << 32) | words[0]; }

vector<u32> bitNeg(const vector<u32> &v) {
  vector<u32> ret;
  ret.reserve(v.size());
  for (auto x : v) { ret.push_back(~x); }
  return ret;
}

static PRPState loadPRP(Gpu *gpu, u32 E, u32 iniB1, u32 iniBlockSize) {
  auto loaded = PRPState::load(E, iniB1, iniBlockSize);    
  gpu->writeState(loaded.check, loaded.base, loaded.blockSize);

  u64 res64 = gpu->dataResidue();
  bool ok = (res64 == loaded.res64);
  gpu->updateCheck();
  log("%s loaded: %d/%d, B1 %u, blockSize %d, %016llx (expected %016llx)\n",
      ok ? "OK" : "EE",  loaded.k, E, loaded.B1, loaded.blockSize, res64, loaded.res64);
  if (!ok) { throw "error on load"; }
  return loaded;
}

vector<u32> Gpu::computeBase(u32 E, u32 B1) {
  u32 nWords = (E - 1) / 32 + 1;
  {
    auto base = vector<u32>(nWords);
    base[0] = 1;    
    this->writeData(base);
  }

  vector<bool> bits = powerSmoothBitsRev(E, B1);
  assert(bits.front());

  Stats stats;
  Timer timer;
    
  u32 k = 0;
  while (k < bits.size()) {
    u32 nIts = min(u32(bits.size() - k), 1000u);
    this->dataLoop(vector<bool>(bits.begin() + k, bits.begin() + (k + nIts)));
    this->finish();
    stats.add(timer.deltaMillis() / float(nIts));
    k += nIts;
    if (k % 10000 == 0) {
      log("   %s\n", makeLogStr(E, k, this->dataResidue(), stats.getStats(), bits.size()).c_str());
      stats.reset();        
    }
  }
  assert(k == bits.size());
  return this->readData();
}

pair<vector<u32>, vector<u32>> Gpu::seedPRP(u32 E, u32 B1) {
  u32 nWords = (E - 1) / 32 + 1;
  
  vector<u32> base;  
  if (B1 == 0) {
    base = vector<u32>(nWords);
    base[0] = 3;
  } else {
    base  = computeBase(E, B1);
    log("Starting P-1 first-stage GCD\n");
    gcd->start(E, base, 1);
  }

  vector<u32> check(nWords);
  check[0] = 1;

  return make_pair(check, base);
}

bool Gpu::isPrimePRP(u32 E, const Args &args, u32 *outB1, u64 *outRes, u64 *outBaseRes, string *outFactor) {
  u32 N = this->getFFTSize();
  log("PRP M(%d), FFT %dK, %.2f bits/word, B1 %u\n", E, N/1024, E / float(N), args.B1);

  if (!PRPState::exists(E)) {
    auto[check, base] = seedPRP(E, args.B1);
    PRPState{0, args.B1, args.blockSize, residue(base), check, base}.save(E);
  }

  PRPState loaded = loadPRP(this, E, args.B1, args.blockSize);
  u32 k = loaded.k;
  u32 B1 = loaded.B1;
  *outB1 = B1;

  u32 blockSize = loaded.blockSize;

  const u32 kEnd = E - 1; // Type-4 per http://www.mersenneforum.org/showpost.php?p=468378&postcount=209
  assert(k < kEnd);

  u32 B2 = args.B2 ? args.B2 : kEnd;
  u32 beginAcc = B2 / 2 / blockSize;
  u32 endAcc   = (B2 - 1) / blockSize + 1;
  
  vector<u32> base = loaded.base;
  
  const u64 baseRes64 = residue(base);
  *outBaseRes = baseRes64;
  assert(blockSize > 0 && 10000 % blockSize == 0);
  const u32 checkStep = blockSize * blockSize;
  
  oldHandler = signal(SIGINT, myHandler);

  int startK = k;
  Stats stats;

  // Number of sequential errors (with no success in between). If this ever gets high enough, stop.
  int nSeqErrors = 0;

  bool isPrime = false;
  Timer timer;

  int nGcdAcc = 0;
  while (true) {
    assert(k % blockSize == 0);
    bool doAcc = (k >= beginAcc * blockSize) && (k < endAcc * blockSize);
    nGcdAcc += doAcc ? blockSize : 0;
    if (kEnd > k && kEnd - k <= blockSize) {
      this->dataLoop(kEnd - k, doAcc);
      auto words = this->roundtripData();
      u64 res64 = residue(words);
      isPrime = (words == base || words == bitNeg(base));

      log("%s %8d / %d, %016llx (base %016llx)\n", isPrime ? "PP" : "CC", kEnd, E, res64, baseRes64);
      
      *outRes = res64;
      *outBaseRes = baseRes64;
      int itersLeft = blockSize - (kEnd - k);
      assert(itersLeft > 0);
      this->dataLoop(itersLeft, doAcc);
      k += blockSize;
    } else {
      do {
        u32 nIters = blockSize;
        assert(nIters > 0);
        this->dataLoop(nIters, doAcc);
        k += nIters;
      } while (k % blockSize != 0);
    }

    u64 res64 = this->dataResidue(); // implies this->finish();

    if (gcd->isReady()) {
      string factor = gcd->get();
      log("GCD says: %s\n", factor.empty() ? "still no factor" : factor.c_str());
      if (!factor.empty()) {
        *outRes = 0;
        *outFactor = factor;
        return false;
      }
    }
        
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
    bool ok = this->doCheck(blockSize);

    bool doSave = k < kEnd && ok;
    if (doSave) { PRPState{k, B1, blockSize, res64, check, base}.save(E); }
    
    doLog(E, k, timer.deltaMillis(), res64, ok, stats);
    
    bool wantGCD = ok && nGcdAcc;
    if (wantGCD) {
      if (gcd->isOngoing()) {
        log("GCD: previous didn't finish\n");
      } else {
        log("Starting GCD over %u Ks\n", nGcdAcc);
        gcd->start(E, this->readAcc(), 0);
        nGcdAcc = 0;
      }
    }

    if (ok) {
      if (isPrime || (k >= kEnd && k >= B2)) { return isPrime; }
      nSeqErrors = 0;
    } else {
      if (++nSeqErrors > 2) {
        log("%d sequential errors, will stop.\n", nSeqErrors);
        throw "too many errors";
      }
      
      auto loaded = loadPRP(this, E, B1, blockSize);
      k = loaded.k;
      assert(blockSize == loaded.blockSize);
      assert(base == loaded.base);
      assert(B1 == loaded.B1);
    }
    if (args.timeKernels) { this->logTimeKernels(); }
    if (doStop) {
      if (gcd->isOngoing()) {
        log("Waiting for GCD to finish..\n");
        gcd->wait();
        if (gcd->isReady()) {
          string factor = gcd->get();
          log("GCD says: %s\n", factor.empty() ? "no factor yet" : factor.c_str());
          if (!factor.empty()) {
            *outRes = 0;
            *outFactor = factor;
            return false;
          }
        }
      }
      throw "stop requested";
    }
  }
}

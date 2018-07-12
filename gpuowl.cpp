// gpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "Gpu.h"
#include "checkpoint.h"
#include "worktodo.h"
#include "args.h"

#include "timeutil.h"
// #include "state.h"
#include "stats.h"
#include "common.h"

#include <cassert>
#include <cstdio>
#include <ctime>
#include <cstdlib>

#include <memory>
#include <string>
#include <vector>
#include <functional>

#include <signal.h>

#define PROGRAM "gpuowl"

static volatile int stopRequested = 0;

void (*oldHandler)(int) = 0;

void myHandler(int dummy) { stopRequested = 1; }

// Residue from compacted words.
u64 residue(const std::vector<u32> &words) { return (u64(words[1]) << 32) | words[0]; }

u32 mod3(const std::vector<u32> &words) {
  u32 r = 0;
  // uses the fact that 2**32 % 3 == 1.
  for (u32 w : words) { r += w % 3; }
  return r % 3;
}

void doDiv3(int E, std::vector<u32> &words) {
  u32 r = (3 - mod3(words)) % 3;
  assert(0 <= r && r < 3);

  int topBits = E % 32;
  assert(topBits > 0 && topBits < 32);

  {
    u64 w = (u64(r) << topBits) + words.back();
    words.back() = w / 3;
    r = w % 3;
  }

  for (auto it = words.rbegin() + 1, end = words.rend(); it != end; ++it) {
    u64 w = (u64(r) << 32) + *it;
    *it = w / 3;
    r = w % 3;
  }
}

// Note: pass vector by copy is intentional.
void doDiv9(int E, std::vector<u32> &words) {
  doDiv3(E, words);
  doDiv3(E, words);
}

string hexStr(u64 res) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%016llx", res);
  return buf;
}

std::string makeLogStr(int E, int k, u64 res, const StatsInfo &info, const string &cpuName) {
  int end = ((E - 1) / 1000 + 1) * 1000;
  float percent = 100 / float(end);
  
  int etaMins = (end - k) * info.mean * (1 / 60000.f) + .5f;
  int days  = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins  = etaMins % 60;
  
  char buf[128];
  snprintf(buf, sizeof(buf), "%s %s%8d/%d [%5.2f%%], %.2f ms/it [%.2f, %.2f]; ETA %dd %02d:%02d; %s",
           shortTimeStr().c_str(),
           cpuName.empty() ? "" : (cpuName + " ").c_str(),
           k, E, k * percent, info.mean, info.low, info.high, days, hours, mins,
           hexStr(res).c_str());
  return buf;
}

void doLog(int E, int k, long timeCheck, u64 res, bool checkOK, int nErrors, Stats &stats, bool didSave, const string &cpuName) {
  std::string errors = !nErrors ? "" : ("; (" + std::to_string(nErrors) + " errors)");
  log("%s %s (check %.2fs)%s%s\n",
      checkOK ? "OK" : "EE",
      makeLogStr(E, k, res, stats.getStats(), cpuName).c_str(),
      timeCheck * .001f, errors.c_str(),
      didSave ? " (saved)" : "");
  stats.reset();
}

void doSmallLog(int E, int k, u64 res, Stats &stats, const string &cpuName) {
  printf("   %s\n", makeLogStr(E, k, res, stats.getStats(), cpuName).c_str());
  stats.reset();
}

// OpenCL or CUDA
extern const char *VARIANT;

bool writeResult(int E, bool isPrime, u64 res, const std::string &AID, const std::string &user, const std::string &cpu, int nErrors, int fftSize) {
  std::string uid;
  if (!user.empty()) { uid += ", \"user\":\"" + user + '"'; }
  if (!cpu.empty())  { uid += ", \"computer\":\"" + cpu + '"'; }
  std::string aidJson = AID.empty() ? "" : ", \"aid\":\"" + AID + '"';
  std::string errors = ", \"errors\":{\"gerbicz\":" + std::to_string(nErrors) + "}";
    
  char buf[512];
  snprintf(buf, sizeof(buf),
           R"-({"exponent":%d, "worktype":"PRP-3", "status":"%c", "residue-type":1, "fft-length":"%dK", "res64":"%s", "program":{"name":"%s", "version":"%s-%s"}, "timestamp":"%s"%s%s%s})-",
           E, isPrime ? 'P' : 'C', fftSize / 1024, hexStr(res).c_str(), PROGRAM, VERSION, VARIANT, timeStr().c_str(),
           errors.c_str(), uid.c_str(), aidJson.c_str());
  
  log("%s\n", buf);
  if (auto fo = open("results.txt", "a")) {
    fprintf(fo.get(), "%s\n", buf);
    return true;
  } else {
    return false;
  }
}

template<typename T, int N> constexpr int size(T (&)[N]) { return N; }

bool isAllZero(const std::vector<u32> &vect) {
  for (const auto x : vect) { if (x) { return false; } }
  return true;
}

bool checkPrime(Gpu *gpu, int E, const Args &args, bool *outIsPrime, u64 *outResidue, int *outNErrors) {
  int k, blockSize, nErrors;  
  u32 N = gpu->getFFTSize();
  
  log("[%s] PRP M(%d), FFT %dK, %.2f bits/word\n", longTimeStr().c_str(), E, N/1024, E / float(N));
  
  {
    LoadResult loaded = Checkpoint::load(E, args.blockSize);
    if (!loaded.ok) {
      log("Invalid checkpoint for exponent %d\n", E);
      return false;
    }
    k = loaded.k;
    blockSize = loaded.blockSize;
    nErrors = loaded.nErrors;
    gpu->writeState(loaded.bits, blockSize);
    u64 res64 = gpu->dataResidue();

    if (loaded.res64) {
      if (res64 == loaded.res64) {
        log("OK loaded: %d/%d, blockSize %d, %016llx\n", k, E, blockSize, res64);
      } else {
        log("EE loaded: %d/%d, blockSize %d, %016llx != %016llx\n", k, E, blockSize, res64, loaded.res64);
        return false;
      }
    }
  }  

  const int checkStep = blockSize * blockSize;
  
  const int kEnd = E; // Residue type-1, see http://www.mersenneforum.org/showpost.php?p=468378&postcount=209
  assert(k % blockSize == 0 && k < kEnd);
  
  oldHandler = signal(SIGINT, myHandler);

  u64 res64 = gpu->dataResidue();
  
  if (gpu->checkAndUpdate(blockSize)) {
    log("OK initial check: %016llx\n", res64);
  } else {
    log("EE initial check: %016llx\n", res64);
    return false;
  }

  gpu->commit();
  int goodK = k;
  int startK = k;
  Stats stats;

  // Number of sequential errors with no success in between. If this ever gets high enough, stop.
  int nSeqErrors = 0;
  
  Timer timer;
  while (true) {
    assert(k % blockSize == 0);

    gpu->dataLoop(std::min(blockSize, kEnd - k));
    
    if (kEnd - k <= blockSize) {
      auto words = gpu->roundtripData();
      u64 resRaw = residue(words);
      doDiv9(E, words);
      u64 resDiv = residue(words);
      words[0] = 0;
      bool isPrime = (resRaw == 9) && isAllZero(words);

      log("%s %8d / %d, %s (raw %s)\n", isPrime ? "PP" : "CC", kEnd, E, hexStr(resDiv).c_str(), hexStr(resRaw).c_str());
      
      *outIsPrime = isPrime;
      *outResidue = resDiv;
      *outNErrors = nErrors;
      int itersLeft = blockSize - (kEnd - k);
      assert(itersLeft > 0);
      gpu->dataLoop(itersLeft);
    }

    gpu->finish();
    k += blockSize;
    auto delta = timer.deltaMillis();
    stats.add(delta * (1/float(blockSize)));

    bool doStop = stopRequested;
    
    if (doStop) {
      log("\nStopping, please wait..\n");
      signal(SIGINT, oldHandler);
    }

    bool doCheck = (k % checkStep == 0) || (k >= kEnd) || doStop || (k - startK == 2 * blockSize);
    if (!doCheck) {
      gpu->updateCheck();
      if (k % 10000 == 0) { doSmallLog(E, k, gpu->dataResidue(), stats, args.cpu); }
      continue;
    }

    u64 res = gpu->dataResidue();
    bool wouldSave = k < kEnd && ((k % 100000 == 0) || doStop);

    // Read GPU state before "check" is updated in gpu->checkAndUpdate().
    std::vector<u32> compactCheck = wouldSave ? gpu->roundtripCheck() : vector<u32>();
    
    bool ok = gpu->checkAndUpdate(blockSize);
    bool doSave = wouldSave && ok;
    if (doSave) {
      Checkpoint::save(E, compactCheck, k, nErrors, blockSize, res);
      
      // just for debug's sake, verify residue match.
      std::vector<u32> compactData = gpu->roundtripData();
      u64 resAux = (u64(compactData[1]) << 32) | compactData[0];
      if (resAux != res) {
        log("Residue mismatch: %016llx %016llx\n", res, resAux);
        return false;
      }
    }
    doLog(E, k, timer.deltaMillis(), res, ok, nErrors, stats, doSave, args.cpu);
    
    if (ok) {
      if (k >= kEnd) { return true; }
      gpu->commit();
      goodK = k;
      nSeqErrors = 0;
    } else {
      if (++nSeqErrors > 10) {
        log("%d sequential errors, will stop.\n", nSeqErrors);
        return false;
      }
      // errorResidue = res;
      ++nErrors;
      gpu->rollback();
      k = goodK;
      // auto offsets = gpu->getOffsets();
      // log("Back to last good iteration %d. Offsets are: data %d, check %d\n", goodK, offsets.first, offsets.second);
      log("Back to last good iteration %d.\n", goodK);
    }
    // if (args.timeKernels) { gpu->logTimeKernels(); }
    if (doStop) { return false; }
  }
}

u32 modInv(u32 a, u32 m) {
  a = a % m;
  for (u32 i = 1; i < m; ++i) {
    if (a * i % m == 1) { return i; }
  }
  assert(false);
}

// selects OpenGpu or CudaGpu.
unique_ptr<Gpu> makeGpu(u32 E, Args &args);

int main(int argc, char **argv) {  
  initLog();

  log("%s-%s %s\n", PROGRAM, VARIANT, VERSION);
  
  Args args;
  if (!args.parse(argc, argv)) { return -1; }

  while (true) {
    char AID[64];
    int E = worktodoReadExponent(AID);
    if (E <= 0) { break; }

    bool isPrime = false;
    u64 residue = 0;
    int nErrors = 0;

    unique_ptr<Gpu> gpu = makeGpu(E, args);

    if (!checkPrime(gpu.get(), E, args, &isPrime, &residue, &nErrors)
        || !writeResult(E, isPrime, residue, AID, args.user, args. cpu, nErrors, gpu->getFFTSize())
        || !worktodoDelete(E)
        || isPrime) {
      break;
    }
  }

  log("\nBye\n");
}


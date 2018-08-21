// gpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "Gpu.h"
#include "TF.h"
#include "checkpoint.h"
#include "worktodo.h"
#include "args.h"
#include "ghzdays.h"

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

std::string makeLogStr(int E, int k, u64 res, const StatsInfo &info, float ghzMsPerIt) {
  int end = ((E - 1) / 1000 + 1) * 1000;
  float percent = 100 / float(end);
  
  int etaMins = (end - k) * info.mean * (1 / 60000.f) + .5f;
  int days  = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins  = etaMins % 60;
  
  char buf[256];
  snprintf(buf, sizeof(buf), "%8d/%d [%5.2f%%], %.2f ms/it [%.2f, %.2f] (%.1f GHz-day/day); ETA %dd %02d:%02d; %s",
           k, E, k * percent, info.mean, info.low, info.high, ghzMsPerIt / info.mean, days, hours, mins,
           hexStr(res).c_str());
  return buf;
}

void doLog(int E, int k, long timeCheck, u64 res, bool checkOK, int nErrors, Stats &stats, bool didSave, float ghzMsPerIt) {
  std::string errors = !nErrors ? "" : ("; (" + std::to_string(nErrors) + " errors)");
  log("%s %s (check %.2fs)%s%s\n",
      checkOK ? "OK" : "EE",
      makeLogStr(E, k, res, stats.getStats(), ghzMsPerIt).c_str(),
      timeCheck * .001f, errors.c_str(),
      didSave ? " (saved)" : "");
  stats.reset();
}

void doSmallLog(int E, int k, u64 res, Stats &stats, float ghzMsPerIt) {
  log("   %s\n", makeLogStr(E, k, res, stats.getStats(), ghzMsPerIt).c_str());
  stats.reset();
}

// OpenCL or CUDA
extern const char *VARIANT;

bool writeResult(const char *part, int E, const char *workType, const char *status, const std::string &AID, const std::string &user, const std::string &cpu) {
  std::string uid;
  if (!user.empty()) { uid += ", \"user\":\"" + user + '"'; }
  if (!cpu.empty())  { uid += ", \"computer\":\"" + cpu + '"'; }
  std::string aidJson = AID.empty() ? "" : ", \"aid\":\"" + AID + '"';
  // std::string errors = ", \"errors\":{\"gerbicz\":" + std::to_string(nErrors) + "}";
    
  char buf[512];
  snprintf(buf, sizeof(buf),
           R"""({"exponent":%d, "worktype":"%s", "status":"%s", "program":{"name":"%s", "version":"%s-%s"}, "timestamp":"%s"%s%s, %s})""",
           E, workType, status, PROGRAM, VERSION, VARIANT, timeStr().c_str(), uid.c_str(), aidJson.c_str(), part);
  
  log("%s\n", buf);
  if (auto fo = open("results.txt", "a")) {
    fprintf(fo.get(), "%s\n", buf);
    return true;
  } else {
    return false;
  }
}

bool writeResultPRP(int E, bool isPrime, u64 res, const string &AID, const string &user, const string &cpu, int nErrors, int fftSize) {
  char buf[256];
  snprintf(buf, sizeof(buf), R"""("residue-type":1, "fft-length":"%dK", "res64":"%s", "errors":{"gerbicz":%d})""",
           fftSize / 1024, hexStr(res).c_str(), nErrors);

  return writeResult(buf, E, "PRP-3", isPrime ? "P" : "C", AID, user, cpu);
}

bool writeResultTF(int E, u64 factor, int bitLo, int bitHi, u64 beginK, u64 endK,
                   const string &AID, const string &user, const string &cpu) {
  bool hasFactor = factor != 0;
  string factorStr = hasFactor ? string(", \"factors\":[") + to_string(factor) + "]" : "";
  
  char buf[256];
  snprintf(buf, sizeof(buf), R"""("bitlo":%d, "bithi":%d, "begink":%llu, "endk":%llu, "rangecomplete":%s%s)""",
           bitLo, bitHi, beginK, endK, hasFactor ? "false" : "true", factorStr.c_str());
           
  return writeResult(buf, E, "TF", hasFactor ? "F" : "NF", AID, user, cpu);
}

template<typename T, int N> constexpr int size(T (&)[N]) { return N; }

bool isAllZero(const std::vector<u32> &vect) {
  for (const auto x : vect) { if (x) { return false; } }
  return true;
}

bool load(Gpu *gpu, u32 E, u32 desiredBlockSize, int *outK, int *outBlockSize, int *outNErrors) {
  LoadResult loaded = Checkpoint::load(E, desiredBlockSize);
  if (!loaded.ok) {
    log("Invalid checkpoint for exponent %d\n", E);
    return false;
  }

  gpu->writeState(loaded.bits, loaded.blockSize);
  u64 res64 = gpu->dataResidue();

  if (loaded.res64 && res64 != loaded.res64) {
    log("EE loaded: %d/%d, blockSize %d, %016llx, expected %016llx\n", loaded.k, E, loaded.blockSize, res64, loaded.res64);
    return false;
  }
  
  log("OK loaded: %d/%d, blockSize %d, %016llx\n", loaded.k, E, loaded.blockSize, res64);
  *outK = loaded.k;
  *outBlockSize = loaded.blockSize;
  *outNErrors = loaded.nErrors;
  return true;
}

bool checkPrime(Gpu *gpu, int E, const Args &args, bool *outIsPrime, u64 *outResidue, int *outNErrors) {
  int k = 0, blockSize = 0, nErrors = 0;
  u32 N = gpu->getFFTSize();
  
  log("PRP M(%d), FFT %dK, %.2f bits/word, %.0f GHz-day\n", E, N/1024, E / float(N), ghzDays(E, N));

  float ghzMsPerIt = ghzSecsPerIt(N) * 1000;

  if (!load(gpu, E, args.blockSize, &k, &blockSize, &nErrors)) { return false; }

  const int checkStep = blockSize * blockSize;
  
  const int kEnd = E; // Residue type-1, see http://www.mersenneforum.org/showpost.php?p=468378&postcount=209
  assert(k % blockSize == 0 && k < kEnd);
  
  oldHandler = signal(SIGINT, myHandler);

  if (gpu->checkAndUpdate(blockSize)) {
    log("OK initial check\n");
  } else {
    log("EE initial check\n");
    return false;
  }

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
      log("Stopping, please wait..\n");
      signal(SIGINT, oldHandler);
    }

    bool doCheck = (k % checkStep == 0) || (k >= kEnd) || doStop || (k - startK == 2 * blockSize);
    if (!doCheck) {
      gpu->updateCheck();
      if (k % 10000 == 0) {
        doSmallLog(E, k, gpu->dataResidue(), stats, ghzMsPerIt);
        if (args.timeKernels) { gpu->logTimeKernels(); }
      }
      continue;
    }

    u64 res = gpu->dataResidue();

    // Read GPU state before "check" is updated in gpu->checkAndUpdate().
    std::vector<u32> compactCheck = gpu->roundtripCheck();
    
    bool ok = gpu->checkAndUpdate(blockSize);
    bool doSave = (k < kEnd) && ok;
    if (doSave) { Checkpoint::save(E, compactCheck, k, nErrors, blockSize, res); }
    
    doLog(E, k, timer.deltaMillis(), res, ok, nErrors, stats, doSave, ghzMsPerIt);
    
    if (ok) {
      if (k >= kEnd) { return true; }
      nSeqErrors = 0;
    } else {
      if (++nSeqErrors > 2) {
        log("%d sequential errors, will stop.\n", nSeqErrors);
        return false;
      }
      if (!load(gpu, E, args.blockSize, &k, &blockSize, &nErrors)) { return false; }
      ++nErrors;
    }
    if (args.timeKernels) { gpu->logTimeKernels(); }
    if (doStop) { return false; }
  }
}

// selects OpenCL or CUDA implementation.
unique_ptr<Gpu> makeGpu(u32 E, Args &args);
unique_ptr<TF> makeTF(Args &args);

// Ideally how far we want an exponent TF-ed.
int targetBits(u32 exp) { return 81 + 2.5 * (log2(exp) - log2(332000000)); }

// Return true if a factor was found.
bool doTF(u32 exp, int bitLo, int bitEnd, Args &args, const string &AID) {
  if (bitLo >= bitEnd) { return false; }
  
  TFState state = Checkpoint::loadTF(exp);
  if (state.bitLo >= bitEnd || (state.nDone == state.nTotal && state.bitHi >= bitEnd)) { return false; }

  if (state.bitLo > bitLo) { bitLo = state.bitLo; }
  
  if (bitLo >= bitEnd) { return false; }

  unique_ptr<TF> tf = makeTF(args);
  assert(tf);

  u64 beginK = 0, endK = 0;
  int nDone = (state.bitHi >= bitEnd) ? state.nDone : 0;
  u64 factor = tf->findFactor(exp, bitLo, bitEnd, nDone, state.nTotal, &beginK, &endK);
  bool ok = writeResultTF(exp, factor, bitLo, bitEnd, beginK, endK, AID, args.user, args.cpu);
  assert(ok);
  return (factor != 0);
}

extern string globalCpuName;

int main(int argc, char **argv) {  
  initLog("gpuowl.log");

  Args args;
  if (!args.parse(argc, argv)) { return -1; }
  if (!args.cpu.empty()) { globalCpuName = args.cpu; }

  log("%s-%s %s\n", PROGRAM, VARIANT, VERSION);
  
  while (true) {
    Task task = Worktodo::getTask();
    if (task.kind == Task::NONE) { break; }

    u32 exp = task.exponent;
    if (task.kind == Task::PRP) {
      if (task.bitLo && TF::enabled() && doTF(exp, task.bitLo, targetBits(exp) + args.tfDelta, args, task.AID)) {
        // If a factor is found by TF, skip and drop the PRP task.
        if (!Worktodo::deleteTask(task)) { break; }
        continue;
      }
      
      bool isPrime = false;
      u64 residue = 0;
      int nErrors = 0;
      unique_ptr<Gpu> gpu = makeGpu(task.exponent, args);

      if (!checkPrime(gpu.get(), exp, args, &isPrime, &residue, &nErrors)
          || !writeResultPRP(exp, isPrime, residue, task.AID, args.user, args.cpu, nErrors, gpu->getFFTSize())
          || !Worktodo::deleteTask(task)
          || isPrime) { // stop on prime found.
        break;
      }
    } else {
      assert(task.kind == Task::TF && task.bitLo < task.bitHi);
      assert(TF::enabled());      
      doTF(exp, task.bitLo, task.bitHi, args, task.AID);
      if (!Worktodo::deleteTask(task)) { break; }
    }
  }

  log("Bye\n");
}


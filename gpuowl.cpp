// gpuOwL, a GPU OpenCL primality tester for Mersenne numbers.
// Copyright (C) 2017 Mihai Preda.

#include "worktodo.h"
#include "args.h"
#include "kernel.h"
#include "timeutil.h"
#include "checkpoint.h"
#include "state.h"
#include "stats.h"
#include "common.h"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <cstdlib>

#include <memory>
#include <string>
#include <vector>
#include <functional>

#include <signal.h>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#define TAU (2 * M_PIl)

// The git revision should be passed through -D on the compiler command line (see Makefile).
#ifndef REV
#define REV
#endif

#define VERSION "1.10-" REV
#define PROGRAM "gpuowl"

static volatile int stopRequested = 0;

void (*oldHandler)(int) = 0;

void myHandler(int dummy) {
  stopRequested = 1;
  // signal(SIGINT, oldHandler);
}

const unsigned BUF_CONST = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
const unsigned BUF_RW    = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;

// Sets the weighting vectors direct A and inverse iA (as per IBDWT).
// FGT doesn't use weight vectors.
template<typename T>
void genWeights(int W, int H, int E, T *aTab, T *iTab) {
  T *pa = aTab;
  T *pi = iTab;

  int N = 2 * W * H;
  int baseBits = E / N;
  auto iN = 1 / (long double) N;

  for (int line = 0; line < H; ++line) {
    for (int col = 0; col < W; ++col) {
      for (int rep = 0; rep < 2; ++rep) {
        int k = (line + col * H) * 2 + rep;
        int bits  = bitlen(N, E, k);
        assert(bits == baseBits || bits == baseBits + 1);
        auto a = exp2l(extra(N, E, k) * iN);
        auto ia = 1 / (4 * N * a);
        *pa++ = (bits == baseBits) ? a  : -a;
        *pi++ = (bits == baseBits) ? ia : -ia;
      }
    }
  }
}

template<typename T> struct Pair { T x, y; };

using double2 = Pair<double>;
using float2  = Pair<float>;
using uint2   = Pair<u32>;
using ulong2  = Pair<u64>;

double2 U2(double a, double b) { return double2{a, b}; }

// Returns the primitive root of unity of order N, to the power k.
template<typename T2> T2 root1(u32 N, u32 k);

template<> double2 root1<double2>(u32 N, u32 k) {
  long double angle = - TAU / N * k;
  double x = cosl(angle);
  double y = sinl(angle);
  return double2{x, y};
}

template<typename T2>
T2 *trig(T2 *p, int n, int B) {
  for (int i = 0; i < n; ++i) { *p++ = root1<T2>(B, i); }
  return p;
}

// The generated trig table has two regions:
// - a H*2 "full trig circle".
// - a region of granularity TAU / (2 * W * H), used in squaring.
cl_mem genSquareTrig(cl_context context, int W, int H) {
  const int size = H * 2 + W;
  auto *tab = new double2[size];
  auto *end = tab;
  end = trig(end, H * 2,     H * 2);
  end = trig(end, W,     W * H * 2);
  assert(end - tab == size);
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double2) * size, tab);
  delete[] tab;
  return buf;
}

// Trig table used by transpose. Has two regions:
// - a size-2048 "full circle",
// - a region of granularity W*H and size W*H/2048.
cl_mem genTransTrig(cl_context context, int W, int H) {
  const int size = 2048 + W * H / 2048;
  auto *tab = new double2[size];
  auto *end = tab;
  end = trig(end, 2048, 2048);
  end = trig(end, W * H / 2048, W * H);
  assert(end - tab == size);
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double2) * size, tab);
  delete[] tab;
  return buf;
}

/*
cl_mem genTrig(cl_context context, int size, int total) {
  assert(size <= total && total % size == 0);

  T2 *tab = new T2[size];
  trig(end, size, total);
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(T2) * size, tab);
  delete[] tab;
  return buf;
}
*/

template<typename T2>
T2 *smallTrigBlock(int W, int H, T2 *p) {
  for (int line = 1; line < H; ++line) {
    for (int col = 0; col < W; ++col) {
      *p++ = root1<T2>(W * H, line * col);
      // if (line == 2 && W == 8) { printf("%4d %4d: %x %x\n", col, line, (u32) p[-1].x, (u32) p[-1].y); }
    }
  }
  return p;
}

cl_mem genSmallTrig(cl_context context, int size, int radix) {
  auto *tab = new double2[size]();
  auto *p = tab + radix;
  int w = 0;
  for (w = radix; w < size; w *= radix) { p = smallTrigBlock(w, radix, p); }
  assert(w == size);
  assert(p - tab == size);
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double2) * size, tab);
  delete[] tab;
  return buf;
}

/*
template<typename T2>
cl_mem genSmallTrig625(cl_context context) {
  int size = 5 * 125;
  T2 *tab = new T2[size]();
  T2 *p   = tab + 5;
  p = smallTrigBlock(  5, 5, p);
  p = smallTrigBlock( 25, 5, p);
  p = smallTrigBlock(125, 5, p);
  assert(p - tab == size);
  
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(T2) * size, tab);
  delete[] tab;
  return buf;
}

template<typename T2>
cl_mem genSmallTrig1K(cl_context context) {
  int size = 4 * 256;
  T2 *tab = new T2[size]();
  T2 *p   = tab + 4;
  p = smallTrigBlock(  4, 4, p);
  p = smallTrigBlock( 16, 4, p);
  p = smallTrigBlock( 64, 4, p);
  p = smallTrigBlock(256, 4, p);
  assert(p - tab == size);

  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(T2) * size, tab);
  delete[] tab;
  return buf;
}

template<typename T2>
cl_mem genSmallTrig4K(cl_context context) {
  int size = 8 * 512;
  T2 *tab = new T2[size]();
  T2 *p   = tab + 8;
  p = smallTrigBlock(  8, 8, p);
  p = smallTrigBlock( 64, 8, p);
  p = smallTrigBlock(512, 8, p);
  assert(p - tab == size);
  
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(T2) * size, tab);
  delete[] tab;
  return buf;
}

template<typename T2>
cl_mem genSmallTrig2K(cl_context context) {
  int size = 4 * 512; // == 8 * 256.
  T2 *tab = new T2[size]();
  T2 *p   = tab + 8;
  p = smallTrigBlock(  8, 8, p);
  p = smallTrigBlock( 64, 8, p);
  p = smallTrigBlock(512, 4, p);
  assert(p - tab == size);
  
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(T2) * size, tab);
  delete[] tab;
  return buf;
}
*/

template<typename T>
void setupWeights(cl_context context, Buffer &bufA, Buffer &bufI, int W, int H, int E) {
  int N = 2 * W * H;
  T *aTab    = new T[N];
  T *iTab    = new T[N];
  
  genWeights(W, H, E, aTab, iTab);
  bufA.reset(makeBuf(context, BUF_CONST, sizeof(T) * N, aTab));
  bufI.reset(makeBuf(context, BUF_CONST, sizeof(T) * N, iTab));
  
  delete[] aTab;
  delete[] iTab;
}

// Residue from compacted words.
u64 residue(const std::vector<u32> &words) { return (u64(words[1]) << 32) | words[0]; }

u32 mod3(std::vector<u32> &words) {
  u32 r = 0;
  // uses the fact that 2**32 % 3 == 1.
  for (u32 w : words) { r += w % 3; }
  return r % 3;
}

void div3(int E, std::vector<u32> &words) {
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
u64 residueDiv9(int E, std::vector<u32> words) {
  div3(E, words);
  div3(E, words);
  return residue(words);
}

std::vector<std::unique_ptr<FILE>> logFiles;

void initLog() {
  logFiles.push_back(std::unique_ptr<FILE>(stdout));
  if (auto fo = open("gpuowl.log", "a")) {
#if defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE)
    setlinebuf(fo.get());
#endif
    logFiles.push_back(std::move(fo));
  }
}

void log(const char *fmt, ...) {
  va_list va;
  for (auto &f : logFiles) {
    va_start(va, fmt);
    vfprintf(f.get(), fmt, va);
    va_end(va);
#if !(defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE))
    fflush(f.get());
#endif
  }
}

string hexStr(u64 res) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%016llx", res);
  return buf;
}

std::string timeStr() {
  time_t t = time(NULL);
  char buf[64];
  // equivalent to: "%F %T"
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", gmtime(&t));
  return buf;
}

std::string timeStr(const std::string &format) {
  time_t t = time(NULL);
  char buf[64];
  strftime(buf, sizeof(buf), format.c_str(), localtime(&t));
  return buf;
}

std::string longTimeStr()  { return timeStr("%Y-%m-%d %H:%M:%S %Z"); }
std::string shortTimeStr() { return timeStr("%H:%M:%S"); }

void doLog(int E, int k, int verbosity, long timeCheck, int nIt, u64 res, bool checkOK, int nErrors, Stats &stats) {
  std::string errors = !nErrors ? "" : (" (" + std::to_string(nErrors) + " errors)");
  int end = ((E - 1) / 1000 + 1) * 1000;
  float percent = 100 / float(end);
  int days = 0, hours = 0, mins = 0;
  float msPerIt = 0;
  
  if (nIt) {
    msPerIt = stats.mean;
    int etaMins = (end - k) * msPerIt * (1 / 60000.f) + .5f;
    days  = etaMins / (24 * 60);
    hours = etaMins / 60 % 24;
    mins  = etaMins % 60;
  }

  if (verbosity == 0 || stats.n < 2) {
    log("%s %8d / %d [%5.2f%%], %.2f ms/it; ETA %dd %02d:%02d; %s [%s]%s\n",
        checkOK ? "OK" : "EE", k, E, k * percent, msPerIt,
        days, hours, mins,
        hexStr(res).c_str(), shortTimeStr().c_str(), errors.c_str());    
  } else {
    log("%s %8d / %d [%5.2f%%], %.2f ms/it [%.2f, %.2f] CV %.1f%%, check %.2fs; ETA %dd %02d:%02d; %s [%s]%s\n",
        checkOK ? "OK" : "EE", k, E, k * percent, msPerIt, stats.min, stats.max, stats.sd() / msPerIt * 100,
        timeCheck / float(1000),
        days, hours, mins,
        hexStr(res).c_str(), shortTimeStr().c_str(), errors.c_str());
  }
}

bool writeResult(int E, bool isPrime, u64 res, const std::string &AID, const std::string &user, const std::string &cpu, int nErrors, int fftSize) {
  std::string uid;
  if (!user.empty()) { uid += ", \"user\":\"" + user + '"'; }
  if (!cpu.empty())  { uid += ", \"computer\":\"" + cpu + '"'; }
  std::string aidJson = AID.empty() ? "" : ", \"aid\":\"" + AID + '"';
  std::string errors = ", \"errors\":{\"gerbicz\":" + std::to_string(nErrors) + "}";
    
  char buf[512];
  snprintf(buf, sizeof(buf),
           R"-({"exponent":%d, "worktype":"PRP-3", "status":"%c", "residue-type":1, "fft-length":"%dK", "res64":"%s", "program":{"name":"%s", "version":"%s"}, "timestamp":"%s"%s%s%s})-",
           E, isPrime ? 'P' : 'C', fftSize / 1024, hexStr(res).c_str(), PROGRAM, VERSION, timeStr().c_str(),
           errors.c_str(), uid.c_str(), aidJson.c_str());
  
  log("%s\n", buf);
  if (auto fo = open("results.txt", "a")) {
    fprintf(fo.get(), "%s\n", buf);
    return true;
  } else {
    return false;
  }
}

void logTimeKernels(std::initializer_list<Kernel *> kerns) {
  std::vector<Kernel *> kvect(kerns);
  std::sort(kvect.begin(), kvect.end(), [](Kernel *a, Kernel *b) { return a->getTime() >= b->getTime(); });
  for (Kernel *k : kvect) {
    u64 time = k->getTime();
    u64 nCall = k->getCalls();
    if (nCall > 100) { log("%-10s : %5.0f us/call  x %5d\n",
                           k->getName().c_str(), time / (float) nCall, (int) nCall); }
    k->resetTime();
  }
}

template<typename T, int N> constexpr int size(T (&)[N]) { return N; }

int autoStep(int nIters, int nErrors, int blockSize) {
  int x = nIters / (100 + nErrors * 1000);
  int steps[] = {1, 2, 5, 10, 20, 50, 100, 200, 500};
  for (int i = 0; i < size(steps) - 1; ++i) {
    if (x < steps[i] * steps[i + 1]) { return std::max(steps[i] * 1000, blockSize * 2); }
  }
  return steps[size(steps) - 1] * 1000;
}

struct GpuState {
  int N;
  cl_context context;
  cl_queue queue;
  Buffer bufDataHolder, bufCheckHolder;
  cl_mem bufData, bufCheck;

  GpuState(int N, cl_context context, cl_queue queue) :
    N(N),
    context(context),
    queue(queue),
    bufDataHolder(makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
    bufCheckHolder(makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
    bufData(bufDataHolder.get()),
    bufCheck(bufCheckHolder.get())
  {
  }

  void writeNoWait(const State &state) {
    assert(N == state.N);
    ::write(queue, false, bufData,  N * sizeof(int), state.data.get());
    ::write(queue, false, bufCheck, N * sizeof(int), state.check.get());
  }

  void writeWait(const State &state) {
    assert(N == state.N);
    ::write(queue, false, bufData, N * sizeof(int), state.data.get());
    ::write(queue, true, bufCheck, N * sizeof(int), state.check.get());
  }

  State read() {
    std::unique_ptr<int[]> data(new int[N]);
    ::read(queue, false, bufData,  N * sizeof(int), data.get());
    
    std::unique_ptr<int[]> check(new int[N]);
    ::read(queue, true,  bufCheck, N * sizeof(int), check.get());
    return State(N, std::move(data), std::move(check));
  }
};

bool checkPrime(int W, int H, int E, cl_queue queue, cl_context context, const Args &args,
                bool *outIsPrime, u64 *outResidue, int *outNErrors, auto modSqLoop, auto modMul,
                std::initializer_list<Kernel *> allKerns) {
  const int N = 2 * W * H;
  log("PRP-3: FFT %dK (%d * %d * 2) of %d (%.2f bits/word) [%s]\n", N / 1024, W, H, E, E / float(N), longTimeStr().c_str());

  int nErrors = 0;
  int k = 0;
  int blockSize = 0;
  State goodState(N);
  
  if (!Checkpoint::load(E, W, H, &goodState, &k, &nErrors, &blockSize)) { return false; }
  log("Starting at iteration %d\n", k);
  
  GpuState gpu(N, context, queue);
  gpu.writeWait(goodState);
  goodState.reset();
  
  int goodK = 0;
  
  const int kEnd = E;
  assert(k % blockSize == 0 && k < kEnd);
  
  auto getCheckStep = [forceStep = args.step, startK = k, startErrors = nErrors, blockSize](int currentK, int currentErrors) {
    return forceStep ? forceStep : autoStep(currentK - startK, currentErrors - startErrors, blockSize);
  };
  
  int blockStartK = k;
  int checkStep = 1; // request an initial check at start.

  // The floating-point transforms use "balanced" words, while the NTT transforms don't.
  const bool balanced = true;
  
  Timer timer;
  Stats stats;
  
  while (true) {
    if (stopRequested) {
      log("\nStopping, please wait..\n");
      signal(SIGINT, oldHandler);
    }

    if ((k % checkStep == 0) || (k >= kEnd) || stopRequested) {
      {
        State state = gpu.read();
        CompactState compact(state, W, H, E);
        compact.expandTo(&state, balanced, W, H, E);
        gpu.writeNoWait(state);
        
        modMul(gpu.bufData, gpu.bufCheck);
        modSqLoop(gpu.bufCheck, blockSize, true);
        bool ok = gpu.read().equalCheck();

        if (ok && k && k < kEnd) {
          Checkpoint::save(compact, k, nErrors, blockSize);
          goodState = std::move(state);
          goodK = k;
        }
        
        doLog(E, k, args.verbosity, timer.deltaMillis(), k - blockStartK, residue(compact.data), ok, nErrors, stats);
        if (args.timeKernels) { logTimeKernels(allKerns); }
        stats.reset();
        
        if (ok) {
          if (k >= kEnd) { return true; }
        } else {        
          assert(k); // A rollback from start (k == 0) means bug or wrong FFT size, so we can't continue.
          ++nErrors;
          k = goodK;
        }
      }
      
      if (stopRequested) { return false; }
      
      gpu.writeNoWait(goodState);
      blockStartK = k;
      checkStep = getCheckStep(k, nErrors);
      assert(checkStep % blockSize == 0);
    }
    
    assert(k % blockSize == 0);

    modMul(gpu.bufCheck, gpu.bufData);
    modSqLoop(gpu.bufData, std::min(blockSize, kEnd - k), false);
    
    if (kEnd - k <= blockSize) {
      State state = gpu.read();
      gpu.writeNoWait(state);      
      std::vector<u32> words = CompactState(state, W, H, E).data;
      
      bool isPrime = (words[0] == 9) && isAllZero(words.begin() + 1, words.end());    
      u64 resRaw = residue(words);
      u64 resDiv = residueDiv9(E, std::move(words));
      log("%s %8d / %d, %s (raw %s)\n", isPrime ? "PP" : "CC", kEnd, E, hexStr(resDiv).c_str(), hexStr(resRaw).c_str());
      
      *outIsPrime = isPrime;
      *outResidue = resDiv;
      *outNErrors = nErrors;
      int left = blockSize - (kEnd - k);
      assert(left > 0);
      modSqLoop(gpu.bufData, left, false);
    }

    finish(queue);
    k += blockSize;
    auto delta = timer.deltaMillis();
    stats.add(delta * (1/float(blockSize)));
    if (args.verbosity >= 2) {
      fprintf(stderr, " %5d / %d, %.2f ms/it    \r", (k - 1) % checkStep + 1, checkStep, delta / float(blockSize));
    }
  }
}

cl_device_id getDevice(const Args &args) {
  cl_device_id device = nullptr;
  if (args.device >= 0) {
    cl_device_id devices[16];
    int n = getDeviceIDs(false, 16, devices);
    assert(n > args.device);
    device = devices[args.device];
  } else {
    int n = getDeviceIDs(true, 1, &device);
    if (n <= 0) {
      log("No GPU device found. See -h for how to select a specific device.\n");
      return 0;
    }
  }
  return device;
}

void append(auto &vect, auto what) { vect.insert(vect.end(), what); }

string valueDefine(const string &key, u32 value) {
  return key + "=" + std::to_string(value) + "u";
}

u32 modInv(u32 a, u32 m) {
  a = a % m;
  for (u32 i = 1; i < m; ++i) {
    if (a * i % m == 1) { return i; }
  }
  assert(false);
}

bool doIt(cl_device_id device, cl_context context, cl_queue queue, const Args &args, const string &AID, int E, int W, int H) {
  assert(W == 625);
  assert(H == 4096);

  int N = 2 * W * H;

  /*
  {
  int sum = 0;
  for (int i = 0; i < N; ++i) {
    sum += bitlen(N, E, i);
  }
  printf("%d\n", sum);
  }
  */
  
  // for (int i = 0; i < 100; ++i) { printf("%d %d\n", i, bitlen(N, E, i)); }
  
  string configName = std::to_string(N / 1024) + "K";
  
  std::vector<string> defines {valueDefine("EXP", E)}; // , valueDefine("WIDTH", W), valueDefine("HEIGHT", H)};

  // append(defines, valueDefine("LOG_NWORDS", std::log2(N)));
  
  string clArgs = args.clArgs;
  if (!args.dump.empty()) { clArgs += " -save-temps=" + args.dump + "/" + configName; }
    
  Holder<cl_program> program;

  bool timeKernels = args.timeKernels;
  
#define LOAD(name, workSize) Kernel name(program.get(), device, queue, workSize, #name, timeKernels)  
  program.reset(compile(device, context, "gpuowl", clArgs, defines, ""));
  if (!program) { return false; }

  LOAD(fftP,   N / 10);
  LOAD(fft625, N / 10);
  LOAD(fft4K,  N / 16);
    
  LOAD(carryA, N / 32);
  LOAD(carryM, N / 32);
  LOAD(carryB, N / 32);

  LOAD(transposeW, 640 * 256);
  LOAD(transposeH, 640 * 256);
    
  LOAD(square,   N / 4);
  LOAD(multiply, N / 4);

  // LOAD(autoConv, N / (8 * 2 * 2));
  LOAD(autoConv, N / 16);
  LOAD(carryConv, N / 10 + 125);

  // LOAD(test, 256, 1);
#undef LOAD
  program.reset();

  /*
  double testData[] = {2, 0, -1, 0, 4, 0, 3, 0, -5, 0};
  Buffer testBuf{makeBuf(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(double) * 10, testData)};
  test.setArg("io", testBuf);
  test();
  ::read(queue, true, testBuf.get(), sizeof(double) * 10, testData);
  for (int i = 0; i < 5; ++i) {
    printf("%f %f\n", testData[2*i], testData[2*i+1]);
  }
  */
  
  Buffer bufTrig625(genSmallTrig(context, 625, 5));
  Buffer bufTrig4K(genSmallTrig(context, 4096, 8));
  Buffer bufTransTrig(genTransTrig(context, 625, 4096));
  Buffer bufSquareTrig(genSquareTrig(context, 625, 4096));
  
  Buffer bufA, bufI;
  setupWeights<double>(context, bufA, bufI, 625, 4096, E);
  
  u32 wordSize = sizeof(double);  
  u32 bufSize = N * wordSize;
  
  Buffer buf1{makeBuf(    context, BUF_RW, bufSize)};
  Buffer buf2{makeBuf(    context, BUF_RW, bufSize)};
  Buffer buf3{makeBuf(    context, BUF_RW, bufSize)};
  Buffer bufCarry{makeBuf(context, BUF_RW, bufSize)}; // could be N/2 as well.

  int *zero = new int[H + 1]();
  Buffer bufReady{makeBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * (H + 1), zero)};
  delete[] zero;

  // Buffer &trigW = (W == 1024) ? bufTrig1K : bufTrig2K;
  // Buffer &trigH = (H == 1024) ? bufTrig1K : bufTrig2K;
  
  fftP.setArg("out", buf1);
  fftP.setArg("A", bufA);
  fftP.setArg("smallTrig", bufTrig625);
  
  fft625.setArg("io", buf1);
  fft625.setArg("smallTrig", bufTrig625);

  fft4K.setArg("io", buf2);
  fft4K.setArg("smallTrig", bufTrig4K);
  
  transposeW.setArg("in",  buf1);
  transposeW.setArg("out", buf2);
  transposeW.setArg("trig", bufTransTrig);

  transposeH.setArg("in",  buf2);
  transposeH.setArg("out", buf1);
  transposeH.setArg("trig", bufTransTrig);

  /*
  double2 *data = new double2[625 * 4096]();
  data[0] = U2(1, 0);
  data[1] = U2(-1, 0);
  data[625] = U2(0, 1);
  data[626] = U2(0, -1);

  data[623] = U2(2, 0);
  data[624] = U2(-2, 0);
  data[625 + 623] = U2(0, 2);
  data[625 + 624] = U2(0, -2);

  data[4094 * 625] = U2(3, 0);
  data[4094 * 625 + 1] = U2(-3, 0);
  data[4095 * 625] = U2(0, 3);
  data[4095 * 625 + 1] = U2(0, -3);

  write(queue, true, buf1.get(), bufSize, data, 0);

  transposeW();

  read(queue, true, buf2.get(), bufSize, data, 0);

  for (int i : {0, 1, 4096, 4097, 623 * 4096, 623 * 4096 + 1, 624 * 4096, 624 * 4096 + 1, 4094, 4095, 4096 + 4094, 4096 + 4095}) {
    printf("%d %f %f\n", i, data[i].x, data[i].y);
  }

  transposeH();
  read(queue, true, buf1.get(), bufSize, data, 0);
  for (int i : {0, 1, 625, 626, 623, 624, 625 + 623, 625 + 624, 4094 * 625, 4094 * 625 + 1, 4095 * 625, 4095 * 625 + 1}) {
    printf("%d %f %f\n", i, data[i].x, data[i].y);
  }
  */
  
  carryA.setArg("in", buf1);
  carryA.setArg("A", bufI);
  carryA.setArg("carryOut", bufCarry);

  carryM.setArg("in", buf1);
  carryM.setArg("A", bufI);
  carryM.setArg("carryOut", bufCarry);

  carryB.setArg("carryIn", bufCarry);
  
  square.setArg("io", buf2);
  square.setArg("bigTrig", bufSquareTrig);
  
  multiply.setArg("io", buf2);
  multiply.setArg("in", buf3);
  multiply.setArg("bigTrig", bufSquareTrig);
  
  autoConv.setArg("io", buf2);
  autoConv.setArg("smallTrig", bufTrig4K);
  autoConv.setArg("bigTrig", bufSquareTrig);
  
  carryConv.setArg("io", buf1);
  carryConv.setArg("carryShuttle", bufCarry);
  carryConv.setArg("ready", bufReady);
  carryConv.setArg("A", bufA);
  carryConv.setArg("iA", bufI);
  carryConv.setArg("smallTrig", bufTrig625);
  
  float bitsPerWord = E / (float) N;
  if (bitsPerWord > 18.6f) { log("Warning: high word size of %.2f bits may result in errors\n", bitsPerWord); }
  
  bool useLongCarry = args.useLongCarry || (bitsPerWord < 13);
  if (useLongCarry) { log("Note: using long (not-fused) carry kernels\n"); }
  bool useLongTail = args.useLongTail;
  if (useLongTail) { log("Note: using long (not-fused) tail kernels\n"); }

  using vfun = std::function<void()>;

  auto carry = useLongCarry ? vfun([&](){ fft625(); carryA(); carryB(); fftP(); }) : vfun([&](){ carryConv(); });
  auto tail  = useLongTail  ? vfun([&](){ fft4K(); square(); fft4K(); })   : vfun([&](){ autoConv(); });
  
  auto entryKerns = [&fftP, &transposeW, &tail, &transposeH](cl_mem in) {
    fftP.setArg("in", in);
    
    fftP();
    transposeW();
    tail();
    transposeH();    
  };

  auto coreKerns = [&]() {
    carry();
    transposeW();
    tail();
    transposeH();
  };

  auto exitKerns = [&fft625, &carryA, &carryM, &carryB](cl_mem out, bool doMul3) {
    (doMul3 ? carryM : carryA).setArg("out", out);
    carryB.setArg("io",  out);
    
    fft625();
    doMul3 ? carryM() : carryA();
    carryB();
  };
  
  // The IBDWT convolution squaring loop with carry propagation, on 'io', done nIters times.
  // Optional multiply-by-3 at the end.
  auto modSqLoop = [&](cl_mem io, int nIters, bool doMul3) {
    assert(nIters > 0);
            
    entryKerns(io);

    // carry args needed for coreKerns.
    carryA.setArg("out", io);
    carryB.setArg("io",  io);

    for (int i = 0; i < nIters - 1; ++i) { coreKerns(); }

    exitKerns(io, doMul3);
  };

  auto directFFT = [&fftP, &transposeW, &fft4K](cl_mem in, cl_mem out) {
    fftP.setArg("in", in);
    transposeW.setArg("out", out);
    fft4K.setArg("io", out);

    fftP();
    transposeW();
    fft4K();
  };
  
  // The modular multiplication io *= in.
  auto modMul = [&](cl_mem io, cl_mem in) {
    directFFT(in, buf3.get());
    directFFT(io, buf2.get());
    multiply(); // input: buf2, buf3; output: buf2.
    fft4K();
    transposeH();
    exitKerns(io, false);
  };

  bool isPrime;
  u64 residue;
  int nErrors = 0;
  if (!checkPrime(W, H, E, queue, context, args, &isPrime, &residue, &nErrors, std::move(modSqLoop), std::move(modMul),
         {&fftP, &fft625, &fft4K, &carryA, &carryM, &carryB, &transposeW, &transposeH, &square, &multiply, &autoConv, &carryConv})) {
    return false;
  }
  
  if (!(writeResult(E, isPrime, residue, AID, args.user, args.cpu, nErrors, N) && worktodoDelete(E))) { return false; }

  if (isPrime) { return false; } // Request stop if a prime is found.
  
  return true;
}

int main(int argc, char **argv) {  
  initLog();
  
  log("gpuOwL v" VERSION " GPU Mersenne primality checker\n");

  oldHandler = signal(SIGINT, myHandler);
  
  Args args;
  
  if (!args.parse(argc, argv)) { return -1; }

  cl_device_id device = getDevice(args);  
  if (!device) { return -1; }

  if (args.cpu.empty()) { args.cpu = getShortInfo(device); }

  std::string info = getLongInfo(device);
  log("%s\n", info.c_str());
  
  Context contextHolder{createContext(device)};
  cl_context context = contextHolder.get();
  Queue queueHolder{makeQueue(device, context)};
  cl_queue queue = queueHolder.get();

  while (true) {
    char AID[64];
    int E = worktodoReadExponent(AID);
    if (E <= 0) { break; }
    
    int W = 625;
    int H = 4096;    
    if (!doIt(device, context, queue, args, AID, E, W, H)) { break; }
  }

  log("\nBye\n");
}


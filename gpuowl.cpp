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

#define VERSION "2.1-" REV
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
  return double2{double(cosl(angle)), double(sinl(angle))};
}

template<typename T2>
T2 *trig(T2 *p, int n, int B) {
  for (int i = 0; i < n; ++i) { *p++ = root1<T2>(B, i); }
  return p;
}

// The generated trig table has two regions:
// - a size-H half-circle.
// - a region of granularity TAU / (2 * W * H), used in squaring.
cl_mem genSquareTrig(cl_context context, int W, int H) {
  const int size = H/2 + W;
  auto *tab = new double2[size];
  auto *end = tab;
  end = trig(end, H/2,     H * 2);
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

template<typename T2>
T2 *smallTrigBlock(int W, int H, T2 *p) {
  for (int line = 1; line < H; ++line) {
    for (int col = 0; col < W; ++col) {
      *p++ = root1<T2>(W * H, line * col);
    }
  }
  return p;
}

cl_mem genSmallTrig(cl_context context, int size, int radix) {
  auto *tab = new double2[size]();
  auto *p = tab + radix;
  int w = 0;
  for (w = radix; w < size; w *= radix) { p = smallTrigBlock(w, std::min(radix, size / w), p); }
  // assert(w == size);
  assert(p - tab == size);
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double2) * size, tab);
  delete[] tab;
  return buf;
}

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

std::string makeLogStr(int E, int k, const StatsInfo &info) {
  int end = ((E - 1) / 1000 + 1) * 1000;
  float percent = 100 / float(end);
  
  int etaMins = (end - k) * info.mean * (1 / 60000.f) + .5f;
  int days  = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins  = etaMins % 60;
  
  char buf[128];
  snprintf(buf, sizeof(buf), "[%s] %8d / %d [%5.2f%%], %.2f ms/it [%.2f, %.2f]; ETA %dd %02d:%02d;",
           shortTimeStr().c_str(),
           k, E, k * percent, info.mean, info.low, info.high, days, hours, mins);
  return buf;
}

void doLog(int E, int k, long timeCheck, u64 res, bool checkOK, int nErrors, Stats &stats) {
  std::string errors = !nErrors ? "" : ("; (" + std::to_string(nErrors) + " errors)");
  log("%s %s %s (check %.2fs)%s\n",
      checkOK ? "OK" : "EE",
      makeLogStr(E, k, stats.getStats()).c_str(), hexStr(res).c_str(),
      timeCheck * .001f, errors.c_str());
  stats.reset();
}

void doSmallLog(int E, int k, Stats &stats) {
  printf("   %s\n", makeLogStr(E, k, stats.getStats()).c_str());
  stats.reset();
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
  struct Info {
    std::string name;
    StatsInfo stats;
  };

  double total = 0;
  std::vector<Info> infos;
  for (Kernel *k : kerns) {
    Info info{k->getName(), k->resetStats()};
    infos.push_back(info);
    total += info.stats.sum;
  }

  std::sort(infos.begin(), infos.end(), [](const Info &a, const Info &b) { return a.stats.sum >= b.stats.sum; });

  for (Info info : infos) {
    StatsInfo stats = info.stats;
    float percent = 100 / total * stats.sum;
    if (percent >= .5f) {
      log("%3.1f%% %-10s : %5.0f [%5.0f, %5.0f] us/call   x %5d calls\n",
          percent, info.name.c_str(), stats.mean, stats.low, stats.high, stats.n);
    }
  }
  log("\n");
}

template<typename T, int N> constexpr int size(T (&)[N]) { return N; }

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
  int nErrors = 0;
  int k = 0;
  int blockSize = 0;
  State goodState(N);

  if (!Checkpoint::load(E, W, H, &goodState, &k, &nErrors, &blockSize)) { return false; }
  log("[%s] PRP M(%d): FFT %dK (%dx%dx2), %.2f bits/word, block %d, at iteration %d\n",
      longTimeStr().c_str(), E, N / 1024, W, H, E / float(N), blockSize, k);
  
  const int kEnd = E;
  assert(k % blockSize == 0 && k < kEnd);
  
  GpuState gpu(N, context, queue);
  gpu.writeNoWait(goodState);

  oldHandler = signal(SIGINT, myHandler);
  
  modMul(gpu.bufData, gpu.bufCheck);
  modSqLoop(gpu.bufCheck, blockSize, true);
  if (!gpu.read().equalCheck()) {
    log("Error at start; will stop\n");
    return false;
  } else {
    log("OK initial check.\n");
  }
  gpu.writeNoWait(goodState);
  
  int goodK = k;
  int startK = k;
  Stats stats;

  u64 errorResidue = 0;  // Residue at the most recent error. Used for persistent-error detection.
  
  Timer timer;
  while (true) {
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
    
    if (stopRequested) {
      log("\nStopping, please wait..\n");
      signal(SIGINT, oldHandler);
    }

    bool doCheck = (k % 100000 == 0) || (k >= kEnd) || stopRequested || (k - startK == 2 * blockSize);
    
    if (doCheck) {
      {
        State state = gpu.read();
        
        CompactState compact(state, W, H, E);
        compact.expandTo(&state, true, W, H, E);
        gpu.writeNoWait(state);

        modMul(gpu.bufData, gpu.bufCheck);
        modSqLoop(gpu.bufCheck, blockSize, true);
        bool ok = gpu.read().equalCheck();

        bool doSave = ok && k < kEnd && ((k % 1'000'000 == 0) || stopRequested);
        
        if (doSave) {
          Checkpoint::save(compact, k, nErrors, blockSize);
          goodState = std::move(state);
          goodK = k;
        }

        u64 res = residue(compact.data);
        doLog(E, k, timer.deltaMillis(), res, ok, nErrors, stats);
        if (args.timeKernels) { logTimeKernels(allKerns); }
        // stats.reset();
        
        if (ok) {
          if (k >= kEnd) { return true; }
        } else { // Error detected.
          if (errorResidue == res) {
            log("Persistent error; will stop.\n");
            return false;
          }
          errorResidue = res;
          ++nErrors;
          k = goodK;
        }
      }
      
      if (stopRequested) { return false; }
      
      gpu.writeNoWait(goodState);
      // blockStartK = k;
      
    } else {
      if (k % 10000 == 0) {
        doSmallLog(E, k, stats);
        // fprintf(stdout, "    %8d / %d, %.2f ms/it    \r", (k - 1) % checkStep + 1, checkStep, delta / float(blockSize));
      }

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
  assert(W == 2048 || W == 4096);
  assert(H == 2048);

  int N = 2 * W * H;
  int hN = N / 2;
  
  string configName = (N % (1024 * 1024)) ? std::to_string(N / 1024) + "K" : std::to_string(N / (1024 * 1024)) + "M";

  int nW = 8;
  int nH = H / 256;
  
  std::vector<string> defines {valueDefine("EXP", E),
      valueDefine("WIDTH", W),
      valueDefine("NW", nW),
      valueDefine("HEIGHT", H),
      valueDefine("NH", nH),
      };

  float bitsPerWord = E / (float) N;

  bool useLongCarry   = args.carry == Args::CARRY_LONG || bitsPerWord < 15;  
  bool useSplitTail = args.tail == Args::TAIL_SPLIT;
  
  log("Note: using %s carry and %s tail kernels\n",
      useLongCarry ? "long" : "short", useSplitTail ? "split" : "fused");

  string clArgs = args.clArgs;
  if (!args.dump.empty()) { clArgs += " -save-temps=" + args.dump + "/" + configName; }
    
  Holder<cl_program> program;

  bool timeKernels = args.timeKernels;
  
#define LOAD(name, workSize) Kernel name(program.get(), device, queue, workSize, #name, timeKernels)  
  program.reset(compile(device, context, "gpuowl", clArgs, defines, ""));
  if (!program) { return false; }

  LOAD(fftP, hN / nW);
  LOAD(fftW, hN / nW);
  LOAD(fftH, hN / nH);
    
  LOAD(carryA, hN / 16);
  LOAD(carryM, hN / 16);
  LOAD(carryB, hN / 16);

  LOAD(transposeW, (W/64) * (H/64) * 256);
  LOAD(transposeH, (W/64) * (H/64) * 256);
    
  LOAD(square,   hN / 2);
  LOAD(multiply, hN / 2);

  LOAD(tailFused, hN / (2 * nH));
  LOAD(carryFused, (hN / nW) + W / nW);
#undef LOAD
  program.reset();
  
  Buffer bufTrigW(genSmallTrig(context, W, nW));
  Buffer bufTrigH(genSmallTrig(context, H, nH));
  Buffer bufTransTrig(  genTransTrig(context, W, H));
  Buffer bufSquareTrig(genSquareTrig(context, W, H));
  
  Buffer bufA, bufI;
  setupWeights<double>(context, bufA, bufI, W, H, E);
  
  u32 wordSize = sizeof(double);  
  u32 bufSize = N * wordSize;
  
  Buffer buf1{makeBuf(    context, BUF_RW, bufSize)};
  Buffer buf2{makeBuf(    context, BUF_RW, bufSize)};
  Buffer buf3{makeBuf(    context, BUF_RW, bufSize)};
  Buffer bufCarry{makeBuf(context, BUF_RW, bufSize)}; // could be N/2 as well.

  int *zero = new int[H]();
  Buffer bufReady{makeBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * H, zero)};
  delete[] zero;
  
  fftP.setArg("out", buf1);
  fftP.setArg("A", bufA);
  fftP.setArg("smallTrig", bufTrigW);
  
  fftW.setArg("io", buf1);
  fftW.setArg("smallTrig", bufTrigW);
  
  fftH.setArg("io", buf2);
  fftH.setArg("smallTrig", bufTrigH);
  
  transposeW.setArg("in",  buf1);
  transposeW.setArg("out", buf2);
  transposeW.setArg("trig", bufTransTrig);

  transposeH.setArg("in",  buf2);
  transposeH.setArg("out", buf1);
  transposeH.setArg("trig", bufTransTrig);
  
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
  
  tailFused.setArg("io", buf2);
  tailFused.setArg("smallTrig", bufTrigH);
  tailFused.setArg("bigTrig", bufSquareTrig);
  
  carryFused.setArg("io", buf1);
  carryFused.setArg("carryShuttle", bufCarry);
  carryFused.setArg("ready", bufReady);
  carryFused.setArg("A", bufA);
  carryFused.setArg("iA", bufI);
  carryFused.setArg("smallTrig", bufTrigW);
    
  using vfun = std::function<void()>;

  auto carry = useLongCarry ? vfun([&](){ fftW(); carryA(); carryB(); fftP(); }) : vfun([&](){ carryFused(); });
  auto tail  = useSplitTail  ? vfun([&](){ fftH(); square(); fftH(); })   : vfun([&](){ tailFused(); });
  
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

  auto exitKerns = [&fftW, &carryA, &carryM, &carryB](cl_mem out, bool doMul3) {
    (doMul3 ? carryM : carryA).setArg("out", out);
    carryB.setArg("io",  out);
    
    fftW();
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

  auto directFFT = [&fftP, &transposeW, &fftH](cl_mem in, cl_mem out) {
    fftP.setArg("in", in);
    transposeW.setArg("out", out);
    fftH.setArg("io", out);

    fftP();
    transposeW();
    fftH();
  };
  
  // The modular multiplication io *= in.
  auto modMul = [&](cl_mem io, cl_mem in) {
    directFFT(in, buf3.get());
    directFFT(io, buf2.get());
    multiply(); // input: buf2, buf3; output: buf2.
    fftH();
    transposeH();
    exitKerns(io, false);
  };

  bool isPrime = false;
  u64 residue = 0;
  int nErrors = 0;
  if (!checkPrime(W, H, E, queue, context, args, &isPrime, &residue, &nErrors, std::move(modSqLoop), std::move(modMul),
                  {&fftP, &fftW, &fftH, &carryA, &carryM, &carryB, &transposeW, &transposeH, &square, &multiply, &tailFused, &carryFused})) {
    return false;
  }
  
  if (!(writeResult(E, isPrime, residue, AID, args.user, args.cpu, nErrors, N) && worktodoDelete(E))) { return false; }

  if (isPrime) { return false; } // Request stop if a prime is found.
  
  return true;
}

int main(int argc, char **argv) {  
  initLog();
  
  log("gpuOwL v" VERSION " GPU Mersenne primality checker\n");
  
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
    
    int W = 2048;
    int H = 2048;
    if (!doIt(device, context, queue, args, AID, E, W, H)) { break; }
  }

  log("\nBye\n");
}


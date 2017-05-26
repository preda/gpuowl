// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#include "clwrap.h"
#include "timeutil.h"
#include "checkpoint.h"
#include "common.h"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cstdlib>

#include <memory>
#include <string>
#include <vector>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

const int EXP_MIN_2M = 20000000, EXP_MAX_2M = 40000000, EXP_MIN_4M = 35000000, EXP_MAX_4M = 78000000;

#define VERSION "0.3"

const char *AGENT = "gpuowl v" VERSION;

const unsigned BUF_CONST = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
const unsigned BUF_RW    = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;

template<typename T>
struct ReleaseDelete {
  using pointer = T;
  
  void operator()(T t) {
    // log("release %s\n", typeid(T).name());
    release(t);
  }
};

template<typename T> using Holder = std::unique_ptr<T, ReleaseDelete<T> >;

using Buffer  = Holder<cl_mem>;
using Context = Holder<cl_context>;
using Queue   = Holder<cl_queue>;

static_assert(sizeof(Buffer) == sizeof(cl_mem));

class Kernel {
  std::string name;
  Holder<cl_kernel> kernel;
  TimeCounter counter;
  int sizeShift;
  bool doTime;
  int extraGroups;

public:
  Kernel(cl_program program, const char *iniName, int iniSizeShift, MicroTimer &timer, bool doTime, int extraGroups = 0) :
    name(iniName),
    kernel(makeKernel(program, name.c_str())),
    counter(&timer),
    sizeShift(iniSizeShift),
    doTime(doTime),
    extraGroups(extraGroups)
  { }

  void setArgs(auto&... args) { ::setArgs(kernel.get(), args...); }
  void setArg(int pos, auto &arg) { ::setArg(kernel.get(), pos, arg); }
  
  const char *getName() { return name.c_str(); }
  void run(cl_queue q, int N) {
    ::run(q, kernel.get(), (N >> sizeShift) + extraGroups * 256);
    if (doTime) {
      finish(q);
      counter.tick();
    }
  }
  u64 getCounter() { return counter.get(); }
  void resetCounter() { counter.reset(); }
  void tick() { counter.tick(); }
};

int wordPos(int N, int E, int p) {
  // assert(N == (1 << 22));
  i64 x = p * (i64) E;
  return (int) (x >> 22) + (bool) (((int) x) << 10);
}

int bitlen(int N, int E, int p) { return wordPos(N, E, p + 1) - wordPos(N, E, p); }

void genBitlen(int W, int H, int E, double *aTab, double *iTab, byte *bitlenTab) {
  double *pa = aTab;
  double *pi = iTab;
  byte   *pb = bitlenTab;

  int N = 2 * W * H;
  int baseBits = E / N;
  auto iN = 1 / (long double) N;

  for (int line = 0; line < H; ++line) {
    for (int col = 0; col < W; ++col) {
      for (int rep = 0; rep < 2; ++rep) {
        int k = (line + col * H) * 2 + rep;
        i64 kE = k * (i64) E;
        auto p0 = kE * iN;
        int b0 = wordPos(N, E, k);
        int b1 = wordPos(N, E, k + 1);
        int bits  = b1 - b0;
        assert(bits == baseBits || bits == baseBits + 1);
        auto a = exp2l(b0 - p0);
        auto ia = 1 / (4 * N * a);
        *pa++ = (bits == baseBits) ? a  : -a;
        *pi++ = (bits == baseBits) ? ia : -ia;
        *pb++ = bits;
      }
    }
  }
}

cl_mem genBigTrig(cl_context context, int W, int H) {
  const int size = (W / 64) * (H / 64) * (2 * 64 * 64);
  double *tab = new double[size];
  auto base = - M_PIl / (W * H / 2);

  double *p = tab;
  for (int gy = 0; gy < H / 64; ++gy) {
    for (int gx = 0; gx < W / 64; ++gx) {
      for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
          auto angle = (gy * 64 + y) * (gx * 64 + x) * base;
          *p++ = cosl(angle);
          *p++ = sinl(angle);
        }
      }
    }
  }

  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double) * size, tab);
  delete[] tab;
  return buf;
}

cl_mem genSin(cl_context context, int W, int H) {
  const int size = 2 * (W / 2) * H;
  double *tab = new double[size]();
  double *p = tab;
  auto base = - M_PIl / (W * H);
  for (int line = 0; line < H; ++line) {
    for (int col = 0; col < (W / 2); ++col) {
      int k = line + (col + ((line == 0) ? 1 : 0)) * H;
      auto angle = k * base;
      *p++ = sinl(angle);
      *p++ = cosl(angle);
    }
  }

  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double) * size, tab);
  delete[] tab;
  return buf;
}

double *smallTrigBlock(int W, int H, double *out) {
  double *p = out;
  for (int line = 1; line < H; ++line) {
    for (int col = 0; col < W; ++col) {
      auto angle = - M_PIl * line * col / (W * H / 2);
      *p++ = cosl(angle);
      *p++ = sinl(angle);
    }
  }
  return p;
}

cl_mem genSmallTrig2K(cl_context context) {
  int size = 2 * 4 * 512;
  double *tab = new double[size]();
  double *p   = tab + 2 * 8;
  p = smallTrigBlock(  8, 8, p);
  p = smallTrigBlock( 64, 8, p);
  p = smallTrigBlock(512, 4, p);
  assert(p - tab == size);
  
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double) * size, tab);
  delete[] tab;
  return buf;
}

cl_mem genSmallTrig1K(cl_context context) {
  int size = 2 * 4 * 256;
  double *tab = new double[size]();
  double *p   = tab + 2 * 4;
  p = smallTrigBlock(  4, 4, p);
  p = smallTrigBlock( 16, 4, p);
  p = smallTrigBlock( 64, 4, p);
  p = smallTrigBlock(256, 4, p);
  assert(p - tab == size);

  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double) * size, tab);
  delete[] tab;
  return buf;
}

bool isAllZero(int *p, int size) {
  for (int *end = p + size; p < end; ++p) { if (*p) { return false; } }
  return true;
}

int &wordAt(int W, int H, int *data, int w) {
  int col  = w / 2 / H;
  int line = w / 2 % H;
  return data[(line * W + col) * 2 + w % 2];
}

bool prevIsNegative(int W, int H, int *data, int p) {
  int N = 2 * W * H;
  assert(0 <= p && p < N);
  for (int i = 0; i < 16; ++i) {
    --p;
    if (p < 0) { p += N; }
    if (int word = wordAt(W, H, data, p)) { return (word < 0); }
  }
  return false;
}

u64 residue(int W, int H, int E, int *data, int offset) {
  int N = 2 * W * H;
  assert(0 <= offset && offset < E);
  int p  = (offset * (i64) N) / E;
  assert(0 <= p && p < N);
  int b0 = wordPos(N, E, p);
  int b1 = wordPos(N, E, p + 1);
  assert(b0 <= offset && offset < b1);
  i64 r = (wordAt(W, H, data, p) - prevIsNegative(W, H, data, p)) >> (offset - b0);
  for (int haveBits = b1 - offset; haveBits < 64; haveBits += bitlen(N, E, p)) {
    ++p;
    if (p >= N) { p -= N; }
    r += (i64) wordAt(W, H, data, p) << haveBits;
  }
  return r;
}

i64 oneShifted(int N, int E, bool forceEvenWord, int offset, int *outWord) {
  if (offset >= E) { offset -= E; }
  assert(0 <= offset && offset < E);
  int p = (offset * (i64) N) / E;
  assert(0 <= p && p < N);
  if (forceEvenWord && (p & 1)) { --p; }
  int b0 = wordPos(N, E, p);
  assert(b0 <= offset);
  *outWord = p;
  return ((i64) 1) << (offset - b0);
}

// 2**k mod m
unsigned twoExpMod(unsigned k, unsigned m) {
  unsigned r = 1;
  for (int bit = 29; bit >= 0; --bit) {
    u64 r2 = r * (u64) r;
    if (k & (1 << bit)) { r2 *= 2; }
    r = r2 % m;
  }
  return r;
}

int offsetAtIter(int E, int offset, int k) {
  unsigned t = twoExpMod(k, E);
  return ((unsigned) offset * (u64) t) % (unsigned) E;
}

FILE *logFiles[3] = {0, 0, 0};

void log(const char *fmt, ...) {
  va_list va;
  for (FILE **pf = logFiles; *pf; ++pf) {
    va_start(va, fmt);
    vfprintf(*pf, fmt, va);
    va_end(va);
  }
}

void doLog(int E, int k, float err, float maxErr, double msPerIter, u64 res) {
  const float percent = 100 / (float) (E - 2);
  int etaMins = (E - 2 - k) * msPerIter * (1 / (double) 60000) + .5;
  int days  = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins  = etaMins % 60;
  
  log("%08d / %08d [%.2f%%], ms/iter: %.3f, ETA: %dd %02d:%02d; %016llx error %g (max %g)\n",
      k, E, k * percent, msPerIter, days, hours, mins, (unsigned long long) res, err, maxErr);
}

int worktodoReadExponent(char *AID) {
  FILE *fi = open("worktodo.txt", "r");
  if (!fi) { return 0; }

  char line[256];
  char kind[32];
  int exp;
  int ret = 0;
  *AID = 0;
  while (true) {
    if (fscanf(fi, "%255s\n", line) < 1) { break; }
    if (((sscanf(line, "%11[^=]=%32[0-9a-fA-F],%d,%*d,%*d", kind, AID, &exp) == 3
          && (!strcmp(kind, "Test") || !strcmp(kind, "DoubleCheck")))
         || sscanf(line, "Test=%d", &exp) == 1)
        && exp >= EXP_MIN_2M && exp <= EXP_MAX_4M) {
      ret = exp;
      break;
    } else {
      log("worktodo.txt line '%s' skipped\n", line);
    }
  }
  fclose(fi);
  return ret;
}

bool worktodoGetLinePos(int E, int *begin, int *end) {
  FILE *fi = open("worktodo.txt", "r");
  if (!fi) { return false; }

  char line[256];
  char kind[32];
  int exp;
  bool ret = false;
  i64 p1 = 0;
  while (true) {
    if (fscanf(fi, "%255s\n", line) < 1) { break; }
    i64 p2 = ftell(fi);
    if (sscanf(line, "%11[^=]=%*32[0-9a-fA-F],%d,%*d,%*d", kind, &exp) == 2 &&
        (!strcmp(kind, "Test") || !strcmp(kind, "DoubleCheck")) &&
        exp == E) {
      *begin = p1;
      *end = p2;
      ret = true;
      break;
    }
    p1 = p2;
  }
  fclose(fi);
  return ret;
}

bool worktodoDelete(int begin, int end) {
  assert(begin >= 0 && end > begin);
  FILE *fi = open("worktodo.txt", "r");
  if (!fi) { return true; }
  char buf[64 * 1024];
  int n = fread(buf, 1, sizeof(buf), fi);
  fclose(fi);
  if (n == sizeof(buf) || end > n) { return false; }
  memmove(buf + begin, buf + end, n - end);
  
  FILE *fo = open("worktodo-tmp.tmp", "w");
  if (!fo) { return false; }
  int newSize = begin + n - end;
  bool ok = (newSize == 0) || (fwrite(buf, newSize, 1, fo) == 1);
  fclose(fo);
  return ok &&
    (rename("worktodo.txt",     "worktodo.bak") == 0) &&
    (rename("worktodo-tmp.tmp", "worktodo.txt") == 0);
}

bool worktodoDelete(int E) {
  int lineBegin, lineEnd;
  return worktodoGetLinePos(E, &lineBegin, &lineEnd) && worktodoDelete(lineBegin, lineEnd);
}

bool writeResult(int E, bool isPrime, u64 residue, const char *AID, int offset) {
  char buf[128];
  snprintf(buf, sizeof(buf), "M( %d )%c, 0x%016llx, offset = %d, n = %dK, %s, AID: %s",
           E, isPrime ? 'P' : 'C', (unsigned long long) residue, offset, 4096, AGENT, AID);
  log("%s\n", buf);
  if (FILE *fo = open("results.txt", "a")) {
    fprintf(fo, "%s\n", buf);
    fclose(fo);
    return true;
  } else {
    return false;
  }
}

void setupExponentBufs(cl_context context, int W, int H, int E, cl_mem *pBufA, cl_mem *pBufI, cl_mem *pBufBitlen) {
  int N = 2 * W * H;
  double *aTab    = new double[N];
  double *iTab    = new double[N];
  byte *bitlenTab = new byte[N];
  
  genBitlen(W, H, E, aTab, iTab, bitlenTab);
  
  *pBufA      = makeBuf(context, BUF_CONST, sizeof(double) * N, aTab);
  *pBufI      = makeBuf(context, BUF_CONST, sizeof(double) * N, iTab);
  *pBufBitlen = makeBuf(context, BUF_CONST, sizeof(byte)   * N, bitlenTab);

  delete[] aTab;
  delete[] iTab;
  delete[] bitlenTab;
}

void run(const std::vector<Kernel *> &kerns, cl_queue q, int N) {
  for (Kernel *k : kerns) { k->run(q, N); }
}

void logTimeKernels(const std::vector<Kernel *> &kerns, int nIters) {
  if (nIters < 2) { return; }
  u64 total = 0;
  for (Kernel *k : kerns) { total += k->getCounter(); }
  const float iIters = 1 / (float) (nIters - 1);
  const float iTotal = 1 / (float) total;
  for (Kernel *k : kerns) {
    u64 c = k->getCounter();
    k->resetCounter();
    log("  %-12s %.1fus, %02.1f%%\n", k->getName(), c * iIters, c * 100 * iTotal);
  }
  log("  %-12s %.1fus\n", "Total", total * iIters);
}

class Iteration {
  int E;
  int k;
  int off;

public:
  Iteration(int E, int k, int off) : E(E), k(k), off(off) {
    assert(0 <= off && off < E);
    assert(0 <= k && k < E - 1);
  }

  constexpr Iteration &operator=(const Iteration &o) = default;
  Iteration(const Iteration &o) = default;

  operator int() { return k; }
  
  Iteration &operator++() {
    off += off;
    if (off >= E) { off -= E; }
    ++k;
    return *this;
  }

  int offset() { return off; }
};

bool checkPrime(int W, int H, cl_queue q,
                const std::vector<Kernel *> &headKerns,
                const std::vector<Kernel *> &tailKerns,
                const std::vector<Kernel *> &coreKerns,
                int E, int logStep, int saveStep, bool doTimeKernels, bool doSelfTest, bool useLegacy,
                cl_mem bufData, cl_mem bufErr,
                Kernel *mega1K, Kernel *carryA,
                bool *outIsPrime, u64 *outResidue,
                int forceOffset, int *outOffset) {
  const int N = 2 * W * H;

  int *data = new int[N + 32];
  int *saveData = new int[N];
  
  std::unique_ptr<int[]> releaseData(data);
  std::unique_ptr<int[]> releaseSaveData(saveData);

  int startK = 0;
  int rootOffset = 0;
  Checkpoint checkpoint(E, W, H);
  if (!doSelfTest && !checkpoint.load(data, &startK, &rootOffset)) { return false; }

  if (startK == 0) {
    if (forceOffset >= E) {
      log("Invalid -offset %d for exponent %d\n", forceOffset, E);
      return false;
    }
    rootOffset = (forceOffset >= 0) ? forceOffset : (rand() % E);
    memset(data, 0, sizeof(int) * N);
    int wordPos = 0;
    int wordVal = oneShifted(N, E, false, rootOffset + 2, &wordPos);
    wordAt(W, H, data, wordPos) = wordVal;
  }
  *outOffset = rootOffset;
  
  memcpy(saveData, data, sizeof(int) * N);
  
  log("LL FFT %dK (%d*%d*2) of %d (%.2f bits/word) offset %d iteration %d\n",
      N / 1024, W, H, E, E / (double) N, rootOffset, startK);
    
  Timer timer;

  const unsigned zero = 0;
  
  write(q, false, bufData, sizeof(int) * N,  data);
  write(q, false, bufErr,  sizeof(unsigned), &zero);

  float maxErr  = 0;  
  u64 res;
  
  Iteration k(E, startK, offsetAtIter(E, rootOffset, startK));
  Iteration saveK(k);
  
  bool isCheck = false;
  bool isRetry = false;

  const int kEnd = doSelfTest ? 20000 : (E - 2);

  std::vector<Kernel *> runKerns = coreKerns;
  if (useLegacy) {
    runKerns = tailKerns;
    runKerns.insert(runKerns.end(), headKerns.begin(), headKerns.end());
  }
  Kernel *coreCarry = useLegacy ? carryA : mega1K;
  
  do {
    int nextLog = std::min((k / logStep + 1) * logStep, kEnd);
    int nIters  = std::max(nextLog - k, 0);
    if (k < nextLog) {
      run(headKerns, q, N);
      if (doTimeKernels) { headKerns[0]->tick(); headKerns[0]->resetCounter(); }
      int wordPos;
      double wordVal;
      for (; k < nextLog - 1;) {
        ++k;
        if (rootOffset != 0) {
          wordVal = - oneShifted(N, E, true, k.offset() + 1, &wordPos);
          wordPos >>= 1;
          coreCarry->setArg(1, wordPos);
          coreCarry->setArg(2, wordVal);
        }
        run(runKerns, q, N);
      }
      ++k;
      wordVal = - oneShifted(N, E, true, k.offset() + 1, &wordPos);
      wordPos >>= 1;
      carryA->setArg(1, wordPos);
      carryA->setArg(2, wordVal);
      run(tailKerns, q, N);
    }
    
    float err = 0;
    read(q,  false, bufErr,  sizeof(float), &err);
    write(q, false, bufErr,  sizeof(unsigned), &zero);
    read(q,  true,  bufData, sizeof(int) * N, data);
    
    res = residue(W, H, E, data, k.offset());

    float msPerIter = timer.delta() / (float) std::max(nIters, 1);
    doLog(E, k, err, std::max(err, maxErr), msPerIter, res);

    if (doTimeKernels) { logTimeKernels(runKerns, nIters); }

    bool isLoop = k < kEnd && data[0] == 2 && isAllZero(data + 1, N - 1);
    
    float MAX_ERR = 0.47f;
    if (err > MAX_ERR || isLoop) {
      const char *problem = isLoop ? "Loop on LL 0...002" : "Error is too large";
      if (!isRetry) {
        log("%s; retrying\n", problem);
        isRetry = true;
        write(q, false, bufData, sizeof(int) * N, saveData);
        k = saveK;
        continue;  // but don't update maxErr yet.
      } else {
        log("%s persists, stopping.\n", problem);
        return false;
      }
    }
    isRetry = false;
        
    if (!isCheck && maxErr > 0 && err > 1.2f * maxErr && logStep >= 1000 && !doSelfTest) {
      log("Error jump by %.2f%%, doing a consistency check.\n", (err / maxErr - 1) * 100);      
      isCheck = true;
      write(q, false, bufData, sizeof(int) * N, saveData);
      k = saveK;
      continue;
    }

    if (isCheck) {
      isCheck = false;
      if (!memcmp(data, saveData, sizeof(int) * N)) {
        log("Consistency checked OK, continuing.\n");
      } else {
        log("Consistency check FAILED, stopping.\n");
        return false;
      }
    }

    if (!doSelfTest) {
      bool doSavePersist = (k / saveStep != (k - logStep) / saveStep);
      checkpoint.save(data, k, doSavePersist, res, rootOffset);
    }

    maxErr = std::max(maxErr, err);
    memcpy(saveData, data, sizeof(int) * N);
    saveK = k;
  } while (k < kEnd);

  *outIsPrime = isAllZero(data, N);
  *outResidue = res;
  return (k >= kEnd);
}

// return false to stop.
bool parseArgs(int argc, char **argv, const char **extraOpts, int *logStep, int *saveStep, int *forceDevice, bool *timeKernels,
               bool *selfTest, bool *useLegacy, int *forceOffset) {
  const int DEFAULT_LOGSTEP = 20000;
  *logStep  = DEFAULT_LOGSTEP;
  *saveStep = 0;
  *timeKernels = false;
  *selfTest    = false;
  *useLegacy   = false;
  *forceOffset = -1;

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
      log("Command line options:\n"
          "-cl \"<OpenCL compiler options>\"\n"
          "    All the cl options must be included in the single argument following -cl\n"
          "    e.g. -cl \"-D LOW_LDS -D NO_ERR -save-temps=tmp/ -O2\"\n"
          "        -save-temps or -save-temps=tmp or -save-temps=tmp/ : save ISA\n"
          "        -D LOW_LDS : use a variant of the amalgamation kernel with lower LDS\n"
          "        -D NO_ERR  : do not compute maximum rounding error\n"
          "-logstep  <N> : to log every <N> iterations (default %d)\n"
          "-savestep <N> : to persist checkpoint every <N> iterations (default 500*logstep == %d)\n"
          "-time kernels : to benchmark kernels (logstep must be > 1)\n"
          "-legacy       : use legacy (old) kernels\n"
          "-offset <N>   : set offset for newly started exponents (otherwise a random offset is used)\n"
          "-selftest     : perform self tests from 'selftest.txt'\n"
          "                Self-test mode does not load/save checkpoints, worktodo.txt or results.txt.\n",
          *logStep, 500 * *logStep);

      log("-device   <N> : select specific device among:\n");
      
      cl_device_id devices[16];
      int ndev = getDeviceIDs(false, 16, devices);
      for (int i = 0; i < ndev; ++i) {
        char info[256];
        getDeviceInfo(devices[i], sizeof(info), info);
        log("    %d : %s\n", i, info);
      }

      log("\nFiles used by gpuOwL:\n"
          "    - worktodo.txt : contains exponents to test \"Test=N\", one per line\n"
          "    - results.txt : contains LL results\n"
          "    - cN.ll : the most recent checkpoint for exponent <N>; will resume from here\n"
          "    - tN.ll : the previous checkpoint, to be used if cN.ll is lost or corrupted\n"
          "    - bN.ll : a temporary checkpoint that is renamed to cN.ll once successfully written\n"
          "    - sN.iteration.residue.ll : a persistent checkpoint at the given iteration\n");

      log("\nThe lines in worktodo.txt must be of one of these forms:\n"
          "Test=70100200\n"
          "Test=3181F68030F6BF3DCD32B77337D5EF6B,70100200,75,1\n"
          "DoubleCheck=3181F68030F6BF3DCD32B77337D5EF6B,70100200,75,1\n"
          "Test=0,70100200,0,0\n");
      
      return false;
    } else if (!strcmp(arg, "-cl")) {
      if (i < argc - 1) {
        *extraOpts = argv[++i];
      } else {
        log("-cl expects options string to pass to CL compiler\n");
        return false;
      }
    } else if (!strcmp(arg, "-logstep")) {
      if (i < argc - 1) {
        *logStep = atoi(argv[++i]);
        if (*logStep <= 0) {
          log("invalid -logstep '%s'\n", argv[i]);
          return false;
        }
      } else {
        log("-logstep expects <N> argument\n");
        return false;
      }
    } else if (!strcmp(arg, "-savestep")) {
      if (i < argc - 1) {
        *saveStep = atoi(argv[++i]);
        if (*saveStep <= 0) {
          log("invalid -savestep '%s'\n", argv[i]);
          return false;
        }
      } else {
        log("-savestep expects <N> argument\n");
        return false;
      }
    } else if(!strcmp(arg, "-time")) {
      if (i < argc - 1 && !strcmp(argv[++i], "kernels")) {
        *timeKernels = true;
      } else {
        log("-time expects 'kernels'\n");
        return false;
      }
    } else if (!strcmp(arg, "-selftest")) {
      *selfTest = true;
    } else if (!strcmp(arg, "-device")) {
      if (i < argc - 1) {
        *forceDevice = atoi(argv[++i]);
        int nDevices = getNumberOfDevices();
        if (*forceDevice < 0 || *forceDevice >= nDevices) {
          log("invalid -device %d (must be between [0, %d]\n", *forceDevice, nDevices - 1);
          return false;
        }        
      } else {
        log("-device expects <N> argument\n");
        return false;
      }
    } else if (!strcmp(arg, "-legacy")) {
      *useLegacy = true;
    } else if (!strcmp(arg, "-offset")) {
      if (i < argc - 1) {
        *forceOffset = atoi(argv[++i]);
        if (*forceOffset < 0) {
          log("invalid -offset '%s'\n", argv[i]);
          return false;
        }
      } else {
        log("-offset expects <offset> argument\n");
        return false;
      }
    } else {
      log("Argument '%s' not understood\n", arg);
      return false;
    }
  }

  if (!*saveStep) { *saveStep = 500 * *logStep; }
  if (*saveStep < *logStep || *saveStep % *logStep) {
    log("It is recommended to use a persist step that is a multiple of the log step (you provided %d, %d)\n", *saveStep, *logStep);
  }
  if (*timeKernels && *logStep == 1) {
    log("Ignoring time kernels because logStep == 1\n");
    *timeKernels = false;
  }

  char buf[32] = {0};
  if (*forceOffset >= 0) { snprintf(buf, sizeof(buf), "-offset %d ", *forceOffset); }
  log("Config: -logstep %d -savestep %d %s%s%s%s-cl \"%s\"\n", *logStep, *saveStep, *timeKernels ? "-time kernels " : "",
      *selfTest ? "-selftest " : "", *useLegacy ? "-legacy " : "", buf, *extraOpts);
  return true;
}

int getNextExponent(bool doSelfTest, u64 *expectedRes, char *AID) {
  if (doSelfTest) {
    static int nRead = 1;
    FILE *fi = open("selftest.txt", "r");
    if (!fi) { return 0; }
    int E;
    for (int i = 0; i < nRead; ++i) {
      if (fscanf(fi, "%d %llx\n", &E, expectedRes) != 2) { E = 0; break; }
    }
    fclose(fi);
    ++nRead;
    return E;
  } else {
    return worktodoReadExponent(AID);
  }
}

int main(int argc, char **argv) {
  logFiles[0] = stdout;
  {
    FILE *logf = open("gpuowl.log", "a");
#ifdef _DEFAULT_SOURCE
    if (logf) { setlinebuf(logf); }
#endif
    logFiles[1] = logf;
  }
  
  time_t t = time(NULL);
  srand(t);
  log("gpuOwL v" VERSION " GPU Lucas-Lehmer primality checker; %s", ctime(&t));

  const char *extraOpts = "";
  int forceDevice = -1;
  int logStep = 0;
  int saveStep = 0;
  int forceOffset = -1;
  bool doTimeKernels = false, doSelfTest = false, useLegacy = false;
  
  if (!parseArgs(argc, argv, &extraOpts, &logStep, &saveStep, &forceDevice, &doTimeKernels, &doSelfTest, &useLegacy, &forceOffset)) {
    return 0;
  }
  
  cl_device_id device;
  if (forceDevice >= 0) {
    cl_device_id devices[16];
    int n = getDeviceIDs(false, 16, devices);
    assert(n > forceDevice);
    device = devices[forceDevice];
  } else {
    int n = getDeviceIDs(true, 1, &device);
    if (n <= 0) {
      log("No GPU device found. See --help for how to select a specific device.\n");
      return 8;
    }
  }
  
  char info[256];
  getDeviceInfo(device, sizeof(info), info);
  log("%s\n", info);
  
  Context contextHolder{createContext(device)};
  cl_context context = contextHolder.get();

  Timer timer;
  MicroTimer microTimer;
  
  cl_program p = compile(device, context, "gpuowl.cl", extraOpts);
  if (!p) { exit(1); }
#define KERNEL(program, name, shift) Kernel name(program, #name, shift, microTimer, doTimeKernels)
  KERNEL(p, fftPremul1K, 3);
  KERNEL(p, transp1K,    5);
  KERNEL(p, fft2K_1K,    4);
  KERNEL(p, csquare2K,   2);
  KERNEL(p, fft2K,       4);
  KERNEL(p, transpose2K, 5);
  KERNEL(p, fft1K,       3);
  KERNEL(p, carryA,      4);
  KERNEL(p, carryB_2K,   4);
#undef KERNEL
  Kernel mega1K(p, "mega1K", 3, microTimer, doTimeKernels, 1);
  
  log("Compile       : %4d ms\n", timer.delta());
  release(p); p = nullptr;
    
  const int W = 1024, H = 2048;
  const int N = 2 * W * H;
  
  Buffer bufTrig1K{genSmallTrig1K(context)};
  Buffer bufTrig2K{genSmallTrig2K(context)};
  Buffer bufBigTrig{genBigTrig(context, W, H)};
  Buffer bufSins{genSin(context, H, W)}; // transposed W/H !

  Buffer buf1{makeBuf(context,     BUF_RW, sizeof(double) * N)};
  Buffer buf2{makeBuf(context,     BUF_RW, sizeof(double) * N)};
  Buffer bufCarry{makeBuf(context, BUF_RW, sizeof(double) * N)}; // could be N/2 as well.

  Buffer bufErr{makeBuf(context,   CL_MEM_READ_WRITE, sizeof(int))};
  Buffer bufData{makeBuf(context,  CL_MEM_READ_WRITE, sizeof(int) * N)};
  int *zero = new int[2049]();
  Buffer bufReady{makeBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * 2049, zero)};
  delete[] zero;
  log("General setup : %4d ms\n", timer.delta());

  Queue queue{makeQueue(device, context)};
  while (true) {
    u64 expectedRes;
    char AID[64];
    int E = getNextExponent(doSelfTest, &expectedRes, AID);
    if (E <= 0) { break; }

    cl_mem pBufA, pBufI, pBufBitlen;
    timer.delta();
    setupExponentBufs(context, W, H, E, &pBufA, &pBufI, &pBufBitlen);
    Buffer bufA(pBufA), bufI(pBufI), bufBitlen(pBufBitlen);
    unsigned baseBitlen = E / N;
    log("Exponent setup: %4d ms\n", timer.delta());

    fftPremul1K.setArgs(bufData, buf1, bufA, bufTrig1K);
    transp1K.setArgs   (buf1,    bufBigTrig);
    fft2K_1K.setArgs   (buf1,    buf2, bufTrig2K);
    csquare2K.setArgs  (buf2,    bufSins);
    fft2K.setArgs      (buf2,    bufTrig2K);
    transpose2K.setArgs(buf2,    buf1, bufBigTrig);
    fft1K.setArgs      (buf1,    bufTrig1K);
    unsigned offsetWord = 0;
    double offsetVal    = -2;
    carryA.setArgs     (baseBitlen, offsetWord, offsetVal, buf1, bufI, bufData, bufCarry, bufErr);
    carryB_2K.setArgs  (bufData, bufCarry, bufBitlen);
    mega1K.setArgs     (baseBitlen, offsetWord, offsetVal, buf1, bufCarry, bufReady, bufErr, bufA, bufI, bufTrig1K);

    std::vector<Kernel *> headKerns {&fftPremul1K, &transp1K, &fft2K_1K, &csquare2K, &fft2K, &transpose2K};
    std::vector<Kernel *> tailKerns {&fft1K, &carryA, &carryB_2K};
    std::vector<Kernel *> coreKerns {&mega1K, &transp1K, &fft2K_1K, &csquare2K, &fft2K, &transpose2K};
    
    bool isPrime;
    u64 residue;
    int offset = 0;
    if (!checkPrime(W, H, queue.get(),
                    headKerns, tailKerns, coreKerns,
                    E, logStep, saveStep, doTimeKernels, doSelfTest, useLegacy,
                    bufData.get(), bufErr.get(),
                    &mega1K, &carryA,
                    &isPrime, &residue,
                    forceOffset, &offset)) {
      break;
    }
    
    if (doSelfTest) {
      if (expectedRes == residue) {
        log("OK %d\n", E);
      } else {
        log("FAIL %d, expected %016llx, got %016llx\n", E, expectedRes, residue);
        break;
      }
    } else {
      if (!(writeResult(E, isPrime, residue, AID, offset) && worktodoDelete(E))) { break; }
    }
  }
    
  log("\nBye\n");
  FILE *f = logFiles[1];
  logFiles[1] = 0;
  if (f) { fclose(f); }
}

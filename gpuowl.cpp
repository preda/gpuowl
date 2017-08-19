// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#include "worktodo.h"
#include "args.h"
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

#define VERSION "0.6"

const char *AGENT = "gpuowl v" VERSION;

const unsigned BUF_CONST = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
const unsigned BUF_RW    = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;

template<typename T>
struct ReleaseDelete {
  using pointer = T;
  
  void operator()(T t) {
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

double *trig(int n, int B, double *out) {
  auto *p = out;
  auto base = - M_PIl / B;
  for (int i = 0; i < n; ++i) {
    auto angle = i * base;
    *p++ = cosl(angle);
    *p++ = sinl(angle);
  }
  return p;
}

cl_mem genBigTrig(cl_context context, int W, int H) {
  assert(W == 1024 && H == 2048);
  const int size = 2 * 3 * 128;
  double *tab = new double[size];
  double *end = tab;
  end = trig(128, 64 * 128 * 128, end);
  end = trig(128, 64 * 128,       end);
  end = trig(128, 64,             end);
  assert(end - tab == size);
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
      int k = line + col * H;
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

cl_mem genCos2K(cl_context context) {
  double *tab = new double[513];
  double *p = tab;
  for (int i = 0; i < 513; ++i) { *p++ = cosl(M_PIl / 1024 * i); }
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double) * 513, tab);
  delete[] tab;
  return buf;
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

/*
cl_mem genTrig1K(cl_context context) {
  int size = 2 * 3 * 256;
  double *tab = new double[size];
  double *p   = tab;
  p = smallTrigBlock(256, 4, p);
  assert(p - tab == size);

  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double) * size, tab);
  delete[] tab;
  return buf;
}
*/

bool isAllZero(int *p, int size) {
  for (int *end = p + size; p < end; ++p) { if (*p) { return false; } }
  return true;
}

int &wordAt(int W, int H, int *data, int w) {
  int col  = w / 2 / H;
  int line = w / 2 % H;
  return data[(line * W + col) * 2 + w % 2];
}

bool prevIsNegative(int W, int H, int *data) {
  int N = 2 * W * H;
  for (int p = N - 1; p >= 0; --p) {
    if (int word = wordAt(W, H, data, p)) { return (word < 0); }
  }
  return false;
}

u64 residue(int W, int H, int E, int *data) {
  int N = 2 * W * H;
  if (isAllZero(data, N)) { return 0; }
  i64 r = - prevIsNegative(W, H, data);
  for (int p = 0, haveBits = 0; haveBits < 64; ++p) {
    r += (i64) wordAt(W, H, data, p) << haveBits;
    haveBits += bitlen(N, E, p);
  }  
  return r;
}

bool isLoop(int *data, int N) {
  // LL loops on 2 and -1 (solutions to x==x^2 - 2)
  return ((data[0] == 2) || (data[0] == -1)) && isAllZero(data + 1, N - 1);
}

std::vector<u32> compactBits(int W, int H, int E, int *data, int *outCarry) {
  std::vector<u32> out;

  int carry = 0;
  u32 outWord = 0;
  int haveBits = 0;
  
  int N = 2 * W * H;
  for (int p = 0; p < N; ++p) {
    int w = wordAt(W, H, data, p) + carry;
    carry = 0;
    int bits = bitlen(N, E, p);
    while (w < 0) {
      w += 1 << bits;
      carry -= 1;
    }
    while (w >= (1 << bits)) {
      w -= 1 << bits;
      carry += 1;
    }
    assert(0 <= w && w < (1 << bits));
    while (bits) {
      assert(haveBits < 32);
      outWord |= w << haveBits;
      if (haveBits + bits >= 32) {
        w >>= (32 - haveBits);
        bits -= (32 - haveBits);
        out.push_back(outWord);
        outWord = 0;
        haveBits = 0;
      } else {
        haveBits += bits;
        bits = 0;
      }
    }
  }
  if (haveBits) {
    out.push_back(outWord);
    haveBits = 0;
  }
  *outCarry = carry;
  return out;
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
  
  log("%08d / %08d [%.2f%%], ms/iter: %.3f, ETA: %dd %02d:%02d; %016llx roundoff %g (max %g)\n",
      k, E, k * percent, msPerIter, days, hours, mins, (unsigned long long) res, err, maxErr);
}

bool writeResult(int E, bool isPrime, u64 residue, const char *AID, const std::string &uid) {
  char buf[256];
  snprintf(buf, sizeof(buf), "%sM( %d )%c, 0x%016llx, offset = 0, n = %dK, %s, AID: %s",
           uid.c_str(), E, isPrime ? 'P' : 'C', (unsigned long long) residue, 4096, AGENT, AID);
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

#ifndef NO_GMP
#include <gmp.h>
#endif

bool jacobiCheck(int W, int H, int E, int *data) {
#ifdef NO_GMP
  return true;
#else
  Timer timer;
  mpz_t compact, mp;
  mpz_inits(compact, mp, nullptr);
  // set mp = 2^E-1
  mpz_ui_pow_ui(mp, 2, E);
  mpz_sub_ui(mp, mp, 1);

  int carry = 0;
  std::vector<u32> cv = compactBits(W, H, E, data, &carry);
  
  // mpz set from vector<u32>: least significant word first, 4 bytes per word, native endianess, 0 bits nails.
  mpz_import(compact, cv.size(), -1, 4, 0, 0, &cv[0]);
  
  carry -= 2;
  if (carry < 0) {
    mpz_sub_ui(compact, compact, -carry);
  } else if (carry > 0) {
    mpz_add_ui(compact, compact, carry);
  }
  int jacobi = mpz_jacobi(compact, mp);
  mpz_clears(compact, mp, nullptr);
  if (jacobi == -1) {
    log("Jacobi check OK (%d ms)\n", timer.delta());
  } else {
    log("Jacobi check FAIL (%d)\n", jacobi);
  }
  return (jacobi == -1);
#endif
}

bool checkPrime(int W, int H, int E, cl_queue q, cl_context context,
                const std::vector<Kernel *> &headKerns,
                const std::vector<Kernel *> &tailKerns,
                const std::vector<Kernel *> &coreKerns,
                const Args &args,
                bool *outIsPrime, u64 *outResidue, auto setBuffers) {
  const int N = 2 * W * H;
  const int dataSize = sizeof(int) * N;
  
  std::unique_ptr<int[]>
    goodDataHolder(new int[N + 32]),
    dataHolder(new int[N + 32]),
    jacobiDataHolder(new int[N + 32]);
  int *goodData = goodDataHolder.get(), *data = dataHolder.get(), *jacobiData = jacobiDataHolder.get();
  
  Buffer bufDataHolder{makeBuf(context, CL_MEM_READ_WRITE, dataSize)};
  Buffer bufErrHolder{makeBuf(context, CL_MEM_READ_WRITE, sizeof(int))};
  cl_mem bufData = bufDataHolder.get(), bufErr = bufErrHolder.get();
  
  int k = 0, jacobiK = -1;
  Checkpoint checkpoint(E, W, H);

  if (!args.selfTest && !checkpoint.load(goodData, &k)) { return false; }
  
  if (k == 0) {
    memset(goodData, 0, dataSize);
    wordAt(W, H, goodData, 0) = 4; // LL start value is 4.
  }
    
  log("LL FFT %dK (%d*%d*2) of %d (%.2f bits/word) iteration %d\n", N / 1024, W, H, E, E / (double) N, k);
    
  Timer timer;

  write(q, false, bufData, dataSize, goodData);
  
  const int kEnd = args.selfTest ? 20000 : (E - 2);

  float err = 0, maxErr = 0, zero = 0;
  int prevK = -1;
  bool isRetry = false;
  setBuffers(bufData, bufErr);
  
  while (true) {
    int nextK = std::min((k / args.logStep + 1) * args.logStep, kEnd);
    if (args.timeKernels) { headKerns[0]->tick(); headKerns[0]->resetCounter(); }

    if (k < nextK) {
      write(q, false, bufErr, sizeof(float), &zero);
      run(headKerns, q, N);
      for (int i = 1; i < nextK - k; ++i) { run(coreKerns, q, N); }
      run(tailKerns, q, N);
    }

    if (prevK < 0) {
      // the first time through, check initial Jacobi symbol and set Jacobi rollback point.
      if (!k || jacobiCheck(W, H, E, goodData)) {
        memcpy(jacobiData, goodData, dataSize);
        jacobiK = k;
      } else {
        log("%08d / %08d : *initial* Jacobi check failed. Restart from an earlier checkpoint\n", k, E);
        return false;
      }
    } else {
      u64 res = residue(W, H, E, goodData);
      float msPerIter = timer.delta() / float(k - prevK);
      maxErr = std::max(err, maxErr);
      doLog(E, k, err, maxErr, msPerIter, res);
      if (args.timeKernels) { logTimeKernels(coreKerns, k - prevK); }
      
      if (!args.selfTest) {
        bool doSavePersist = (k == kEnd) || (k / args.saveStep  != (k - args.logStep) / args.saveStep);
        bool doJacobiCheck = (k == kEnd) || (k / args.checkStep != (k - args.logStep) / args.checkStep);
        
        if (doJacobiCheck) {
          if (jacobiCheck(W, H, E, goodData)) {
            // set Jacobi rollback point to current.
            memcpy(jacobiData, goodData, dataSize);
            jacobiK = k;
          } else {
            log("%08d / %08d : Jacobi check failed: rolling back to iteration %08d\n", k, E, jacobiK);
          
            //rollback
            finish(q);
            memcpy(goodData, jacobiData, dataSize);
            write(q, false, bufData, dataSize, goodData);
            prevK = -1;
            k = jacobiK;
            continue;
          }
        }
        
        checkpoint.save(goodData, k, doSavePersist, res);
      }
    }

    if (k >= nextK) { break; }

    read(q, false, bufErr, sizeof(float), &err);
    read(q, true, bufData, sizeof(int) * N, data);

    // fprintf(stderr, "-- %d %f\n", k, err2);
    
    bool doRetry = false;
    char mes[64] = {0};
    
    if (isLoop(data, N)) {
      snprintf(mes, sizeof(mes), "loop detected");
      doRetry = true;
    } else if (err > 0.44f) {
      snprintf(mes, sizeof(mes), "roundoff %f is too large", err);
      doRetry = true;
    } 

    if (doRetry) {
      u64 res = residue(W, H, E, data);
      log("%08d / %08d %016llx; Error (will retry) : %s\n", nextK, E, u64(res), mes);
      
      if (isRetry) {
        log("Retry failed to recover, will exit\n");
        return false;
      }

      // Retry. Do not update k & prevK.
      write(q, false, bufData, dataSize, goodData);
    } else {
      std::swap(goodData, data);
      prevK = k;
      k = nextK;
    }
    isRetry = doRetry;
  }

  *outIsPrime = isAllZero(goodData, N);
  *outResidue = residue(W, H, E, goodData);
  return (k >= kEnd);
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

  Args args;
  
  if (!args.parse(argc, argv)) { return 0; }
  
  cl_device_id device;
  if (args.device >= 0) {
    cl_device_id devices[16];
    int n = getDeviceIDs(false, 16, devices);
    assert(n > args.device);
    device = devices[args.device];
  } else {
    int n = getDeviceIDs(true, 1, &device);
    if (n <= 0) {
      log("No GPU device found. See -h for how to select a specific device.\n");
      return 8;
    }
  }
  
  char info[256];
  getDeviceInfo(device, sizeof(info), info);
  log("%s\n", info);
  
  Context contextHolder{createContext(device)};
  cl_context context = contextHolder.get();

  /*
  cl_program test = compile(device, context, "test.cl", "-save-temps=test/", false);
  assert(test);
  printf("test compiled\n");
  assert(dumpBinary(test, "test.bin"));
  printf("bin dumped\n");
  cl_program test2 = compile(device, context, "test.bin", "", false, true);
  assert(test2);
  printf("test2 compiled\n");
  assert(dumpBinary(test2, "test2.bin"));
  printf("bin2 dumped\n");
  exit(0);
  */
  
  Timer timer;
  MicroTimer microTimer;
  
  cl_program p = compile(device, context, "gpuowl.cl", args.clArgs.c_str(), true);
  if (!p) { exit(1); }
#define KERNEL(program, name, shift) Kernel name(program, #name, shift, microTimer, args.timeKernels)
  KERNEL(p, fftPremul1K, 3);
  KERNEL(p, transpose2K, 5);
  KERNEL(p, transpose1K, 5);
  KERNEL(p, fft1K,       3);
  KERNEL(p, carryA,      4);
  KERNEL(p, carryB_2K,   4);
  KERNEL(p, tail,        5);
#undef KERNEL
  Kernel mega(p, "mega", 3, microTimer, args.timeKernels, 1);
  
  log("Compile       : %4d ms\n", timer.delta());
  release(p); p = nullptr;
    
  const int W = 1024, H = 2048;
  const int N = 2 * W * H;
  
  Buffer bufTrig1K{genSmallTrig1K(context)};
  Buffer bufTrig2K{genSmallTrig2K(context)};
  Buffer bufCos2K{genCos2K(context)};
  Buffer bufBigTrig{genBigTrig(context, W, H)};
  Buffer bufSins{genSin(context, H, W)}; // transposed W/H !

  Buffer buf1{makeBuf(context,     BUF_RW, sizeof(double) * N)};
  Buffer buf2{makeBuf(context,     BUF_RW, sizeof(double) * N)};
  Buffer bufCarry{makeBuf(context, BUF_RW, sizeof(double) * N)}; // could be N/2 as well.

  int *zero = new int[2049]();
  Buffer bufReady{makeBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * 2049, zero)};
  delete[] zero;
  log("General setup : %4d ms\n", timer.delta());

  Queue queue{makeQueue(device, context)};
  
  while (true) {
    u64 expectedRes;
    char AID[64];
    int E = getNextExponent(args.selfTest, &expectedRes, AID);
    if (E <= 0) { break; }

    cl_mem pBufA, pBufI, pBufBitlen;
    timer.delta();
    setupExponentBufs(context, W, H, E, &pBufA, &pBufI, &pBufBitlen);
    Buffer bufA(pBufA), bufI(pBufI), bufBitlen(pBufBitlen);
    unsigned baseBitlen = E / N;
    log("Exponent setup: %4d ms\n", timer.delta());

    Buffer dummy;
    
    fftPremul1K.setArgs(dummy, buf1, bufA, bufTrig1K);
    tail.setArgs       (buf2,    bufTrig2K, bufSins);
    transpose2K.setArgs(buf2,    buf1, bufBigTrig);
    transpose1K.setArgs(buf1,    buf2, bufBigTrig);
    fft1K.setArgs      (buf1,    bufTrig1K);
    carryA.setArgs     (baseBitlen, buf1, bufI, dummy, bufCarry, dummy);
    carryB_2K.setArgs  (dummy, bufCarry, bufBitlen);
    mega.setArgs(baseBitlen, buf1, bufCarry, bufReady, dummy, bufA, bufI, bufTrig1K);
    
    bool isPrime;
    u64 residue;

    std::vector<Kernel *> headKerns {&fftPremul1K, &transpose1K, &tail, &transpose2K};
    std::vector<Kernel *> tailKerns {&fft1K, &carryA, &carryB_2K};    
    std::vector<Kernel *> coreKerns {&mega, &transpose1K, &tail, &transpose2K};
    if (args.useLegacy) {
      coreKerns = tailKerns;
      coreKerns.insert(coreKerns.end(), headKerns.begin(), headKerns.end());
    }

    if (!checkPrime(W, H, E, queue.get(), context,
                    headKerns, tailKerns, coreKerns,
                    args,
                    &isPrime, &residue,
                    [&fftPremul1K, &carryA, &carryB_2K, &mega](cl_mem data, cl_mem err) {
                      fftPremul1K.setArg(0, data);
                      carryA.setArg(3, data);
                      carryA.setArg(5, err);
                      carryB_2K.setArg(0, data);
                      mega.setArg(4, err);
                    }
                    )) {
      break;
    }
    
    if (args.selfTest) {
      if (expectedRes == residue) {
        log("OK %d\n", E);
      } else {
        log("FAIL %d, expected %016llx, got %016llx\n", E, expectedRes, residue);
        break;
      }
    } else {
      std::string uid = args.uid.empty() ? "" : "UID: " + args.uid + ", ";
      if (!(writeResult(E, isPrime, residue, AID, uid) && worktodoDelete(E))) { break; }
    }
  }
    
  log("\nBye\n");
  FILE *f = logFiles[1];
  logFiles[1] = 0;
  if (f) { fclose(f); }
}

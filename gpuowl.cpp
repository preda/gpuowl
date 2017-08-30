// gpuOwL, a GPU OpenCL primality tester for Mersenne numbers.
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

#define VERSION "1.0"

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

  template<int P> void setArgsAt() {}  
  template<int P> void setArgsAt(auto &a, auto&... args) {
    setArg(P, a);
    setArgsAt<P + 1>(args...);
  }
  
public:
  Kernel(cl_program program, const char *iniName, int iniSizeShift, MicroTimer &timer, bool doTime, int extraGroups = 0) :
    name(iniName),
    kernel(makeKernel(program, name.c_str())),
    counter(&timer),
    sizeShift(iniSizeShift),
    doTime(doTime),
    extraGroups(extraGroups)
  { }

  void setArg(int pos, const Buffer &buf) { setArg(pos, buf.get()); }
  void setArg(int pos, const auto &arg) { ::setArg(kernel.get(), pos, arg); } 
  void setArgs(const auto&... args) { setArgsAt<0>(args...); }

  
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

// Sets the weighting vectors direct A and inverse iA (as per IBDWT).
void genWeights(int W, int H, int E, double *aTab, double *iTab) {
  double *pa = aTab;
  double *pi = iTab;

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

bool isAllZero(const int *p, int size) {
  for (const int *end = p + size; p < end; ++p) { if (*p) { return false; } }
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

u32 crc32(const void *data, size_t size) {
  u32 tab[16] = {
    0x00000000, 0x1DB71064, 0x3B6E20C8, 0x26D930AC,
    0x76DC4190, 0x6B6B51F4, 0x4DB26158, 0x5005713C,
    0xEDB88320, 0xF00F9344, 0xD6D6A3E8, 0xCB61B38C,
    0x9B64C2B0, 0x86D3D2D4, 0xA00AE278, 0xBDBDF21C,
  };
  u32 crc = ~0;
  for (auto *p = (const unsigned char *) data, *end = p + size; p < end; ++p) {
    crc = tab[(crc ^  *p      ) & 0xf] ^ (crc >> 4);
    crc = tab[(crc ^ (*p >> 4)) & 0xf] ^ (crc >> 4);
  }
  return ~crc;
}

void doLog(int E, int k, float msPerIter, u64 res, bool checkOK) {
  int end = ((E - 1) / 1000 + 1) * 1000;
  const float percent = 100 / float(end);
  int etaMins = (end - k) * msPerIter * (1 / (float) 60000) + .5f;
  int days  = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins  = etaMins % 60;

  char buf[64];
  snprintf(buf, sizeof(buf), "P3-%d-%d-%016llx", E, k, res);
  u32 crc = crc32(buf, strlen(buf));
  
  log("%s %05dK (%05.2f%%) of %d, %.2f ms/it, ETA %dd %02d:%02d; P3-%016llx%02x\n",
      checkOK ? "OK" : "EE", k / 1000, k * percent, E, msPerIter, days, hours, mins, res, crc & 0xff);
}

bool writeResult(int E, bool isPrime, u64 res, const char *AID, const std::string &uid) {
  char buf[256];
  snprintf(buf, sizeof(buf), "P3-%d-%d-%016llx", E, E-1, res);
  u32 crc = crc32(buf, strlen(buf));
  
  snprintf(buf, sizeof(buf), "%sM( %d )%c, P3-%016llx%02x, n = %dK, %s, AID: %s",
           uid.c_str(), E, isPrime ? 'P' : 'C', res, crc & 0xff, 4096, AGENT, AID);
  log("%s\n", buf);
  if (FILE *fo = open("results.txt", "a")) {
    fprintf(fo, "%s\n", buf);
    fclose(fo);
    return true;
  } else {
    return false;
  }
}

void setupExponentBufs(cl_queue q, int W, int H, int E, cl_mem bufA, cl_mem bufI) {
  int N = 2 * W * H;
  double *aTab    = new double[N];
  double *iTab    = new double[N];
  
  genWeights(W, H, E, aTab, iTab);
  write(q, false, bufA, sizeof(double) * N, aTab);
  write(q, true, bufI, sizeof(double) * N, iTab);

  delete[] aTab;
  delete[] iTab;
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

bool validate(int N, cl_mem bufData, cl_mem bufCheck,
              cl_queue q, auto squareLoop, auto modMul,
              const int *data, const int *check) {
  const int dataSize = sizeof(int) * N;
  
  if (isAllZero(data, N)) { return false; }
  
  Timer timer;
  write(q, false, bufCheck, dataSize, check);
  write(q, false, bufData,  dataSize, data);
  modMul(q, bufData, bufCheck);
  squareLoop(q, bufCheck, 1000, true);
  int *tmpA(new int[N]), *tmpB(new int[N]);
  read(q, false, bufData, dataSize, tmpA);
  read(q, true, bufCheck, dataSize, tmpB);
  bool ok = !memcmp(tmpA, tmpB, dataSize);
  delete[] tmpA;
  delete[] tmpB;
  write(q, false, bufCheck, dataSize, check);
  write(q, false, bufData, dataSize, data);
  // log("Check %d (%d ms)\n", int(ok), timer.delta());
  return ok;
}

bool checkPrime(int W, int H, int E, cl_queue q, cl_context context, const Args &args,
                bool *outIsPrime, u64 *outResidue, auto modSqLoop, auto modMul) {
  const int N = 2 * W * H;
  const int dataSize = sizeof(int) * N;

  std::unique_ptr<int[]>
    goodDataHolder(new int[N]),
    goodCheckHolder(new int[N]),
    dataHolder(new int[N]),
    checkHolder(new int[N]);
  int *goodData  = goodDataHolder.get();
  int *goodCheck = goodCheckHolder.get();
  int *data  = dataHolder.get();
  int *check = checkHolder.get();
  
  Buffer bufDataHolder{makeBuf(context, CL_MEM_READ_WRITE, dataSize)};
  Buffer bufCheckHolder{makeBuf(context, CL_MEM_READ_WRITE, dataSize)};
  
  cl_mem bufData  = bufDataHolder.get();
  cl_mem bufCheck = bufCheckHolder.get();
  
  int k = 0, goodK = 0;
  Checkpoint checkpoint(E, W, H);

  if (!checkpoint.load(&k, data, check)) { return false; }
  log("PRP-3 FFT %dK (%d*%d*2) of %d (%.2f bits/word) iteration %d\n", N / 1024, W, H, E, E / (double) N, k);
  assert(k % 1000 == 0);
  const int kEnd = E - 1;
  assert(k < kEnd);
  
  auto setRollback = [=, &goodK, &k]() {
    memcpy(goodData,  data,  dataSize);
    memcpy(goodCheck, check, dataSize);
    goodK = k;
  };

  auto rollback = [=, &goodK, &k]() {
    log("rolling back to %d\n", goodK);
    write(q, false, bufData, dataSize, goodData);
    write(q, false, bufCheck, dataSize, goodCheck);
    k = goodK;
  };

  if (k == 0) {
    memset(data,  0, dataSize);
    memset(check, 0, dataSize);
    data[0]  = 3;
    check[0] = 1;    
  }
  
  Timer timer;

  // Establish a known-good roolback point by initial verification of loaded data.
  while (true) {
    bool ok = validate(N, bufData, bufCheck, q, modSqLoop, modMul, data, check);
    doLog(E, k, timer.delta() / 1000.f, residue(W, H, E, data), ok);
    if (ok) { break; }
    assert(k > 0);
    log("Loaded checkpoint failed validation. Restarting from zero.\n");
    k = 0;
    memset(data,  0, dataSize);
    memset(check, 0, dataSize);
    data[0]  = 3;
    check[0] = 1;        
  }
  setRollback();

  while (k < kEnd) {
    assert(k % 1000 == 0);
    modMul(q, bufCheck, bufData);    
    modSqLoop(q, bufData, std::min(1000, kEnd - k), false);
    if (kEnd - k <= 1000) {
      read(q, true, bufData, dataSize, data);
      // The write() below may seem redundant, but it protects against memory errors on the read() above,
      // by making sure that any eventual errors are visible to the GPU-side verification.
      write(q, false, bufData, dataSize, data);

      *outResidue = residue(W, H, E, data);
      *outIsPrime = data[0] == -3 && isAllZero(data + 1, N - 1);
        
      int left = 1000 - (kEnd - k);
      assert(left >= 0);
      if (left) { modSqLoop(q, bufData, left, false); }
    }

    finish(q);
    k += 1000;
    fprintf(stderr, " %2.0f%%\r", k % args.step * (100.f / args.step));

    if ((k % args.step == 0) || (k >= kEnd)) {
      read(q, false, bufCheck, dataSize, check);
      read(q, true, bufData, dataSize, data);
      bool ok = validate(N, bufData, bufCheck, q, modSqLoop, modMul, data, check);
      doLog(E, k, timer.delta() / float(args.step), residue(W, H, E, data), ok);      
      if (ok) {
        setRollback();
        if (k < kEnd) { checkpoint.save(k, data, check, k % args.saveStep == 0); }
      } else {
        rollback();
      }
    }
  }
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
  log("gpuOwL v" VERSION " GPU Mersenne primality checker; %s", ctime(&t));

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

  KERNEL(p, fft2K,     4);
  KERNEL(p, csquare2K, 2);
  KERNEL(p, cmul2K,    2);
  KERNEL(p, carryMul3,  4);
#undef KERNEL
  Kernel mega(p, "mega", 3, microTimer, args.timeKernels, 1);
  
  log("Compile       : %4d ms\n", timer.delta());
  release(p); p = nullptr;
    
  constexpr int W = 1024, H = 2048;
  constexpr int N = 2 * W * H;
  
  Buffer bufTrig1K{genSmallTrig1K(context)};
  Buffer bufTrig2K{genSmallTrig2K(context)};
  Buffer bufCos2K{genCos2K(context)};
  Buffer bufBigTrig{genBigTrig(context, W, H)};
  Buffer bufSins{genSin(context, H, W)}; // transposed W/H !

  Buffer buf1{makeBuf(context, BUF_RW, sizeof(double) * N)};
  Buffer buf2{makeBuf(context, BUF_RW, sizeof(double) * N)};
  Buffer buf3{makeBuf(context, BUF_RW, sizeof(double) * N)};
  Buffer bufCarry{makeBuf(context, BUF_RW, sizeof(double) * N)}; // could be N/2 as well.

  // Weights (direct and inverse) for the IBDWT.
  Buffer bufA{makeBuf(context, CL_MEM_READ_ONLY, sizeof(double) * N)};
  Buffer bufI{makeBuf(context, CL_MEM_READ_ONLY, sizeof(double) * N)};
  
  int *zero = new int[2049]();
  Buffer bufReady{makeBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * 2049, zero)};
  delete[] zero;

  Buffer dummy;
    
  fftPremul1K.setArgs(dummy, buf1, bufA, bufTrig1K);
  tail.setArgs       (buf2,    bufTrig2K, bufSins);
  transpose2K.setArgs(buf2,    buf1, bufBigTrig);
  transpose1K.setArgs(buf1,    buf2, bufBigTrig);
  fft1K.setArgs      (buf1,    bufTrig1K);
  carryA.setArgs     (0, buf1, bufI, dummy, bufCarry);
  carryMul3.setArgs  (0, buf1, bufI, dummy, bufCarry);
  carryB_2K.setArgs  (0, dummy, bufCarry, bufI);
  mega.setArgs(0, buf1, bufCarry, bufReady, bufA, bufI, bufTrig1K);

  fft2K.setArgs(buf2, bufTrig2K);
  csquare2K.setArgs(buf2, bufSins);
  cmul2K.setArgs(buf2, buf3, bufSins);

  log("General setup : %4d ms\n", timer.delta());

  Queue queue{makeQueue(device, context)};
  
  while (true) {
    u64 expectedRes;
    char AID[64];
    int E = getNextExponent(false, &expectedRes, AID);
    if (E <= 0) { break; }

    setupExponentBufs(queue.get(), W, H, E, bufA.get(), bufI.get());
    unsigned baseBitlen = E / N;
    carryA.setArg(0, baseBitlen);
    carryMul3.setArg(0, baseBitlen);
    carryB_2K.setArg(0, baseBitlen);
    mega.setArg(0, baseBitlen);
    
    bool isPrime;
    u64 residue;

    // std::vector<Kernel *> headKerns {&fftPremul1K, &transpose1K, &tail, &transpose2K};

    // the weighting + direct FFT only, stops before square/mul.
    std::vector<Kernel *> directFftKerns {&fftPremul1K, &transpose1K, &fft2K};

    // sequence of: direct FFT, square, first-half of inverse FFT.
    std::vector<Kernel *> headKerns(directFftKerns);
    headKerns.insert(headKerns.end(), { &csquare2K, &fft2K, &transpose2K });

    // sequence of: second-half of inverse FFT, inverse weighting, carry propagation.
    std::vector<Kernel *> tailKerns {&fft1K, &carryA, &carryB_2K};

    // kernel-fusion equivalent of: tailKerns, headKerns.
    // std::vector<Kernel *> coreKerns {&mega, &transpose1K, &fft2K, &csquare2K, &fft2K, &transpose2K};
    std::vector<Kernel *> coreKerns {&mega, &transpose1K, &tail, &transpose2K};
    if (args.useLegacy) {
      coreKerns = tailKerns;
      coreKerns.insert(coreKerns.end(), headKerns.begin(), headKerns.end());
    }

    // The IBDWT convolution squaring loop with carry propagation, on 'data', done nIters times.
    // Optional multiply-by-3 at the end.
    auto modSqLoop = [&](cl_queue q, cl_mem data, int nIters, bool doMul3) {
      assert(nIters > 0);
            
      if (args.timeKernels) { headKerns[0]->tick(); headKerns[0]->resetCounter(); }

      fftPremul1K.setArg(0, data);
      run(headKerns, q, N);

      carryA.setArg(3, data);
      carryB_2K.setArg(1, data);

      for (int i = 0; i < nIters - 1; ++i) { run(coreKerns, q, N); }
      if (doMul3) {
        carryMul3.setArg(3, data);
        run({&fft1K, &carryMul3, &carryB_2K}, q, N);
      } else {
        run(tailKerns, q, N);
      }

      if (args.timeKernels) { logTimeKernels(coreKerns, nIters); }
    };

    // The modular multiplication a = a * b. Output in 'a'.
    auto modMul = [&](cl_queue q, cl_mem a, cl_mem b) {
      fftPremul1K.setArg(0, b);
      transpose1K.setArg(1, buf3);
      fft2K.setArg(0, buf3);
      run(directFftKerns, q, N);
      
      fftPremul1K.setArg(0, a);
      transpose1K.setArg(1, buf2);
      fft2K.setArg(0, buf2);
      run(directFftKerns, q, N);
      
      run({&cmul2K, &fft2K, &transpose2K}, q, N);

      carryA.setArg(3, a);
      carryB_2K.setArg(1, a);
      run(tailKerns, q, N);
    };

    if (!checkPrime(W, H, E, queue.get(), context, args, &isPrime, &residue, std::move(modSqLoop), std::move(modMul))) {
      break;
    }

    std::string uid = args.uid.empty() ? "" : "UID: " + args.uid + ", ";
    if (!(writeResult(E, isPrime, residue, AID, uid) && worktodoDelete(E))) { break; }
  }
    
  log("\nBye\n");
  FILE *f = logFiles[1];
  logFiles[1] = 0;
  if (f) { fclose(f); }
}

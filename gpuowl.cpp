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
#include <algorithm>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#define TAU (2 * M_PIl)

#define VERSION "1.5"
#define PROGRAM "gpuowl"

const unsigned BUF_CONST = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
const unsigned BUF_RW    = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;

template<typename T>
struct ReleaseDelete {
  using pointer = T;
  
  void operator()(T t) {
    // fprintf(stderr, "Release %s %llx\n", typeid(T).name(), u64(t));
    release(t);
  }
};

template<typename T> using Holder = std::unique_ptr<T, ReleaseDelete<T> >;

using Buffer  = Holder<cl_mem>;
using Context = Holder<cl_context>;
using Queue   = Holder<cl_queue>;

static_assert(sizeof(Buffer) == sizeof(cl_mem));

class Kernel {  
  template<int P> void setArgsAt() {}  
  template<int P> void setArgsAt(auto &a, auto&... args) {
    setArg(P, a);
    setArgsAt<P + 1>(args...);
  }

public:
  virtual ~Kernel() {};
  virtual void run(cl_queue q) = 0;
  virtual string getName() = 0;
  virtual cl_kernel getKernel() = 0;
  
  void setArg(int pos, const auto &arg) { ::setArg(getKernel(), pos, arg); }
  void setArg(int pos, const Buffer &buf) { setArg(pos, buf.get()); }
  void setArgs(const auto&... args) { setArgsAt<0>(args...); }
};

class BaseKernel : public Kernel {
  Holder<cl_kernel> kernel;
  int N;
  int sizeShift;
  std::string name;

public:
  BaseKernel(cl_program program, int N, const std::string &name, int sizeShift) :
    kernel(makeKernel(program, name.c_str())),
    N(N),
    sizeShift(sizeShift),
    name(name)
  { }

  virtual void run(cl_queue q) { ::run(q, kernel.get(), N >> sizeShift); }

  virtual string getName() { return name; }

  virtual cl_kernel getKernel() { return kernel.get(); }
};

class TimedKernel : public Kernel {
  std::unique_ptr<Kernel> kernel;
  u64 timeAcc;

public:
  TimedKernel(Kernel *k) : kernel(k), timeAcc(0) { }

  virtual void run(cl_queue q) {
    Timer timer;
    kernel->run(q);
    finish(q);
    timeAcc += timer.deltaMicros();
  }

  virtual string getName() { return kernel->getName(); }
  virtual cl_kernel getKernel() { return kernel->getKernel(); }
  
  u64 getTime() { return timeAcc; }
  void resetTime() { timeAcc = 0; }
};

int wordPos(int N, int E, int p) {
  assert(N == (1 << 21) || N == (1 << 22) || N == (1 << 23));
  
  i64 x = p * (i64) E;
  
  return
    (N == (1 << 21)) ? (int) (x >> 21) + (bool) (((int) x) << 11) :
    (N == (1 << 22)) ? (int) (x >> 22) + (bool) (((int) x) << 10) :
    (int) (x >> 23) + (bool) (((int) x) << 9);
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

double *trig(double *p, int n, int B, int phase = 0, int step = 1) {
  auto base = - TAU / B;
  for (int i = 0; i < n; ++i) {
    auto angle = (phase + i * step) * base;
    *p++ = cosl(angle);
    *p++ = sinl(angle);
  }
  return p;
}

// The generated trig table has three regions:
// - a 4096 "full trig table" (a full circle).
// - a region of granularity TAU / (W * H), used in transpose.
// - a region of granularity TAU / (2 * W * H), used in squaring.
cl_mem genBigTrig(cl_context context, int W, int H) {
  assert((W == 1024 || W == 2048) && (H == 1024 || H == 2048));
  const int size = 2 * (H * 2 + W / 2 + W);
  double *tab = new double[size];
  double *end = tab;
  end = trig(end, H * 2, H * 2);
  end = trig(end, W / 2, W * H);
  end = trig(end, W,     W * H * 2);
  assert(end - tab == size);
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double) * size, tab);
  delete[] tab;
  return buf;
}

double *smallTrigBlock(int W, int H, double *p) {
  auto base = - TAU / (W * H);
  for (int line = 1; line < H; ++line) {
    for (int col = 0; col < W; ++col) {
      auto angle = line * col * base;
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

void setupWeights(cl_context context, Buffer &bufA, Buffer &bufI, int W, int H, int E) {
  int N = 2 * W * H;
  double *aTab    = new double[N];
  double *iTab    = new double[N];
  
  genWeights(W, H, E, aTab, iTab);
  bufA.reset(makeBuf(context, BUF_CONST, sizeof(double) * N, aTab));
  bufI.reset(makeBuf(context, BUF_CONST, sizeof(double) * N, iTab));
  
  delete[] aTab;
  delete[] iTab;
}

bool isAllZero(auto const it, auto const end) {
  return std::all_of(it, end, [](auto e) { return e == 0; });
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

// Residue from compacted words.
u64 residue(const std::vector<u32> &words) { return (u64(words[1]) << 32) | words[0]; }

// Residue from balanced irrational-base words.
u64 residue(int W, int H, int E, int *data) {
  int N = 2 * W * H;
  if (isAllZero(data, data + N)) { return 0; }
  i64 r = - prevIsNegative(W, H, data);
  for (int p = 0, haveBits = 0; haveBits < 64; ++p) {
    r += (i64) wordAt(W, H, data, p) << haveBits;
    haveBits += bitlen(N, E, p);
  }  
  return r;
}

std::vector<u32> compactBits(int W, int H, int E, int *data) {
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

  for (int p = 0; carry; ++p) {
    i64 v = i64(out[p]) + carry;
    out[p] = v & 0xffffffff;
    carry = v >> 32;
  }

  return out;
}

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

std::vector<FILE *> logFiles;

void log(const char *fmt, ...) {
  va_list va;
  for (FILE *f : logFiles) {
    va_start(va, fmt);
    vfprintf(f, fmt, va);
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

u32 checksum(int E, int k, u64 res) {
  char buf[64];
  snprintf(buf, sizeof(buf), "P3-%d-%d-%016llx", E, k, res);
  return crc32(buf, strlen(buf));
} 

string hexStr(u64 res) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%016llx", res);
  return buf;
}

std::string timeStr() {
  time_t t = time(NULL);
  char buf[64];
  strftime(buf, sizeof(buf), "%F %T", gmtime(&t));
  return buf;
}

std::string localTimeStr() {
  time_t t = time(NULL);
  char buf[64];
  strftime(buf, sizeof(buf), "%F %T %Z", localtime(&t));
  return buf;
}

void doLog(int E, int k, float msPerIter, u64 res, bool checkOK, int nErrors) {
  int end = ((E - 1) / 1000 + 1) * 1000;
  const float percent = 100 / float(end);
  int etaMins = (end - k) * msPerIter * (1 / (float) 60000) + .5f;
  int days  = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins  = etaMins % 60;

  std::string errors = !nErrors ? "" : (" (" + std::to_string(nErrors) + " errors)");
  
  log("%s %8d / %d [%5.2f%%], %.2f ms/it, ETA %dd %02d:%02d; %s [%s]%s\n",
      checkOK ? "OK" : "EE", k, E, k * percent, msPerIter, days, hours, mins,
      hexStr(res).c_str(), localTimeStr().c_str(), errors.c_str());
}

bool writeResult(int E, bool isPrime, u64 res, const std::string &AID, const std::string &user, const std::string &cpu, int nErrors) {
  std::string uid;
  if (!user.empty()) { uid += ", \"user\":\"" + user + '"'; }
  if (!cpu.empty())  { uid += ", \"cpu\":\"" + cpu + '"'; }
  std::string aidJson = AID.empty() ? "" : ", \"aid\":\"" + AID + '"';
  std::string errors = ", \"errors\":{\"gerbicz\":" + std::to_string(nErrors) + "}";
    
  char buf[512];
  snprintf(buf, sizeof(buf),
           R"-({"exponent":%d, "worktype":"PRP-3", "status":"%c", "res64":"%s", "residue-checksum":"%08x", "program":{"name":"%s", "version":"%s"}, "timestamp":"%s"%s%s%s})-",
           E, isPrime ? 'P' : 'C', hexStr(res).c_str(), checksum(E, E-1, res), PROGRAM, VERSION, timeStr().c_str(),
           errors.c_str(), uid.c_str(), aidJson.c_str());

  log("%s\n", buf);
  if (FILE *fo = open("results.txt", "a")) {
    fprintf(fo, "%s\n", buf);
    fclose(fo);
    return true;
  } else {
    return false;
  }
}

void run(const std::vector<Kernel *> &kerns, cl_queue q) { for (Kernel *k : kerns) { k->run(q); } }

void logTimeKernels(const std::vector<Kernel *> &kerns, int nIters) {
  if (nIters < 2) { return; }
  u64 total = 0;
  for (Kernel *k : kerns) { total += dynamic_cast<TimedKernel *>(k)->getTime(); }
  const float iIters = 1 / (float) (nIters - 1);
  const float iTotal = 1 / (float) total;
  for (Kernel *kk : kerns) {
    TimedKernel *k = dynamic_cast<TimedKernel *>(kk);
    u64 c = k->getTime();
    k->resetTime();
    log("%4d us, %02d%% : %s\n", int(c * iIters + .5f), int(c * 100 * iTotal + .5f), k->getName().c_str());
  }
  log("%4d us total\n", int(total * iIters + .5f));
}

bool validate(int N, cl_mem bufData, cl_mem bufCheck,
              cl_queue q, auto squareLoop, auto modMul,
              const int *data, const int *check) {
  const int dataSize = sizeof(int) * N;
  
  if (isAllZero(data, data + N)) { return false; }
  
  Timer timer;
  write(q, false, bufCheck, dataSize, check);
  write(q, false, bufData,  dataSize, data);
  modMul(q, bufData, bufCheck);
  squareLoop(q, bufCheck, 1000, true);
  int *tmpA(new int[N]), *tmpB(new int[N]);
  read(q, false, bufData, dataSize, tmpA);
  read(q, true, bufCheck, dataSize, tmpB);
  bool ok = !memcmp(tmpA, tmpB, dataSize);
  // fprintf(stderr, "%d %d\n", tmpA[0], tmpB[0]);
  delete[] tmpA;
  delete[] tmpB;
  write(q, false, bufCheck, dataSize, check);
  write(q, true, bufData, dataSize, data);
  return ok;
}

bool checkPrime(int W, int H, int E, cl_queue q, cl_context context, const Args &args,
                bool *outIsPrime, u64 *outResidue, int *outNErrors, auto modSqLoop, auto modMul) {
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
  int nErrors = 0;
  Checkpoint checkpoint(E);

  if (!checkpoint.load(W, H, &k, data, check, &nErrors)) { return false; }
  log("PRP-3 FFT %dK (%d*%d*2) of %d (%.2f bits/word) iteration %d\n", N / 1024, W, H, E, E / (double) N, k);
  assert(k % 1000 == 0);
  const int kEnd = E;
  assert(k < kEnd);
  
  auto setRollback = [=, &goodK, &k]() {
    memcpy(goodData,  data,  dataSize);
    memcpy(goodCheck, check, dataSize);
    goodK = k;
  };

  auto rollback = [=, &goodK, &k]() {
    log("rolling back to %d\n", goodK);
    write(q, false, bufData, dataSize, goodData);
    write(q, true, bufCheck, dataSize, goodCheck);
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
    doLog(E, k, timer.deltaMillis() / 1000.f, residue(W, H, E, data), ok, nErrors);
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
    Timer smallTimer;
    modMul(q, bufCheck, bufData);    
    modSqLoop(q, bufData, std::min(1000, kEnd - k), false);
    if (kEnd - k <= 1000) {
      read(q, true, bufData, dataSize, data);
      // The write() below may seem redundant, but it protects against memory errors on the read() above,
      // by making sure that any eventual errors are visible to the GPU-side verification.
      write(q, false, bufData, dataSize, data);

      std::vector<u32> words = compactBits(W, H, E, data);
      bool isPrime = (words[0] == 9) && isAllZero(words.begin() + 1, words.end());    
      u64 resRaw = residue(words);      
      div3(E, words);
      div3(E, words);
      u64 res = residue(words);
      
      log("%s %8d / %d, %s (raw %s)\n", isPrime ? "PP" : "CC", kEnd, E, hexStr(res).c_str(), hexStr(resRaw).c_str());
      
      *outIsPrime = isPrime;
      *outResidue = res;
      *outNErrors = nErrors;
      int left = 1000 - (kEnd - k);
      assert(left >= 0);
      if (left) { modSqLoop(q, bufData, left, false); }
    }

    finish(q);
    k += 1000;
    fprintf(stderr, " %5d / %d, %.2f ms/it\r", (k - 1) % args.step + 1, args.step, smallTimer.deltaMillis() / 1000.f);

    if ((k % args.step == 0) || (k >= kEnd)) {
      read(q, false, bufCheck, dataSize, check);
      read(q, true, bufData, dataSize, data);
      bool ok = validate(N, bufData, bufCheck, q, modSqLoop, modMul, data, check);
      doLog(E, k, timer.deltaMillis() / float((k - 1) % args.step + 1), residue(W, H, E, data), ok, nErrors);      
      if (ok) {
        setRollback();
        if (k < kEnd) { checkpoint.save(W, H, k, data, check, k % args.saveStep == 0, nErrors); }
      } else {
        rollback();
        ++nErrors;
      }
    }
  }

  finish(q); // Redundant. Queue must be empty before buffers release.
  return true;
}

void initLog() {
  logFiles.push_back(stdout);
  if (FILE *logf = open("gpuowl.log", "a")) {
#ifdef _DEFAULT_SOURCE
    setlinebuf(logf);
#endif
    logFiles.push_back(logf);
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

bool doIt(cl_device_id device, cl_context context, cl_queue queue, const Args &args, const string &AID, int E, int W, int H) {
  assert(W == 1024 || W == 2048);
  assert(H == 1024 || H == 2048);
  assert(W <= H);
  int N = 2 * W * H;
  
  std::unique_ptr<Kernel> fftP, fftW, fftH, transpose, transposeT, carryA, carryB, carryM, square, mul, carryConv, squareConv;

  {
    unsigned baseBitlen = E / N;
    std::vector<string> defines{string("BASE_BITLEN=") + std::to_string(baseBitlen)};
    
    if (W == 1024) { append(defines, "GPUOWL_1K"); }
    if (H == 2048) { append(defines, "GPUOWL_2K"); }

    switch (N / (1024 * 1024)) {
    case 2:
      append(defines, "GPUOWL_2M");
      break;
    case 4:
      append(defines, "GPUOWL_4M");
      break;
    case 8:
      append(defines, "GPUOWL_8M");
      break;
    };

    cl_program p = compile(device, context, "gpuowl.cl", args.clArgs, defines);
    Holder<cl_program> programHolder(p);    
    if (!p) { return false; }

    if (W == 1024) {
      fftP.reset(new BaseKernel(p, N, "fftPremul1K", 3));
      fftW.reset(new BaseKernel(p, N, "fft1K",       3));
      
      carryA.reset(new BaseKernel(p, N, "carryA1K",   5));
      carryM.reset(new BaseKernel(p, N, "carryMul1K", 5));
    } else {
      fftP.reset(new BaseKernel(p, N, "fftPremul2K", 4));
      fftW.reset(new BaseKernel(p, N, "fft2K",       4));

      carryA.reset(new BaseKernel(p, N, "carryA2K",   5));
      carryM.reset(new BaseKernel(p, N, "carryMul2K", 5));
    }

    if (H == 1024) {
      fftH.reset(new BaseKernel(p, N, "fft1K", 3));
    } else {
      fftH.reset(new BaseKernel(p, N, "fft2K", 4));
    }

    switch (N / (1024 * 1024)) {
    case 2:
      transpose.reset( new BaseKernel(p, N, "transpose1K_1K", 5));
      transposeT.reset(new BaseKernel(p, N, "transpose1K_1K", 5));

      carryB.reset(new BaseKernel(p, N, "carryB1K_1K", 5));

      carryConv.reset( new BaseKernel(p, N + (256 << 3), "carryConv1K_1K", 3));
      squareConv.reset(new BaseKernel(p, N, "tail1K_1K", 4));

      square.reset(new BaseKernel(p, N, "csquare1K_1K", 2));
      mul.reset(   new BaseKernel(p, N, "cmul1K_1K",    2));
      break;

    case 4:
      transpose.reset( new BaseKernel(p, N, "transpose1K_2K", 5));
      transposeT.reset(new BaseKernel(p, N, "transpose2K_1K", 5));

      carryB.reset(new BaseKernel(p, N, "carryB1K_2K", 5));

      carryConv.reset(new BaseKernel(p, N + (256 << 3), "carryConv1K_2K", 3));
      squareConv.reset(new BaseKernel(p, N, "tail2K_1K", 5));

      square.reset(new BaseKernel(p, N, "csquare2K_1K", 2));
      mul.reset(   new BaseKernel(p, N, "cmul2K_1K",    2));
      break;

    case 8:
      transpose.reset( new BaseKernel(p, N, "transpose2K_2K", 5));
      transposeT.reset(new BaseKernel(p, N, "transpose2K_2K", 5));

      carryB.reset(new BaseKernel(p, N, "carryB2K_2K", 5));

      carryConv.reset(new BaseKernel(p, N + (256 << 4), "carryConv2K_2K", 4));
      squareConv.reset(new BaseKernel(p, N, "tail2K_2K", 5));

      square.reset(new BaseKernel(p, N, "csquare2K_2K", 2));
      mul.reset(   new BaseKernel(p, N, "cmul2K_2K",    2));
      break;

    default:
      assert(false);
    }    

    if (args.timeKernels) {
      for (auto k : {&fftP, &fftW, &fftH, &transpose, &transposeT, &carryA, &carryB, &carryM, &square, &mul, &carryConv, &squareConv}) {
        k->reset(new TimedKernel(k->release()));
      }
    }
  }

  Buffer bufTrig1K{genSmallTrig1K(context)};
  Buffer bufTrig2K{genSmallTrig2K(context)};
  cl_mem trigW = (W == 1024) ? bufTrig1K.get() : bufTrig2K.get();
  cl_mem trigH = (H == 1024) ? bufTrig1K.get() : bufTrig2K.get();

  Buffer bufBigTrig{genBigTrig(context, W, H)};
  
  Buffer buf1{makeBuf(context, BUF_RW, sizeof(double) * N)};
  Buffer buf2{makeBuf(context, BUF_RW, sizeof(double) * N)};
  Buffer buf3{makeBuf(context, BUF_RW, sizeof(double) * N)};
  Buffer bufCarry{makeBuf(context, BUF_RW, sizeof(double) * N)}; // could be N/2 as well.

  // Weights (direct and inverse) for the IBDWT.
  Buffer bufA, bufI;
  setupWeights(context, bufA, bufI, W, H, E);
  
  int *zero = new int[H + 1]();
  Buffer bufReady{makeBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * (H + 1), zero)};
  delete[] zero;

  Buffer dummy;
  
  fftP->setArgs(dummy, buf1, bufA, trigW);
  fftW->setArgs(buf1, trigW);
  fftH->setArgs(buf2, trigH);
  
  transpose->setArgs(buf1, buf2, bufBigTrig);
  transposeT->setArgs(buf2, buf1, bufBigTrig);
  
  carryA->setArgs(buf1, bufI, dummy, bufCarry);
  carryM->setArgs(buf1, bufI, dummy, bufCarry);
  carryB->setArgs(dummy, bufCarry, bufI);

  square->setArgs(buf2, bufBigTrig);
  mul->setArgs(buf2, buf3, bufBigTrig);

  carryConv->setArgs(buf1, bufCarry, bufReady, bufA, bufI, trigW);
  squareConv->setArgs(buf2, trigH, bufBigTrig);

  // the weighting + direct FFT only, stops before square/mul.
  std::vector<Kernel *> directFftKerns {fftP.get(), transpose.get(), fftH.get()};

  // sequence of: direct FFT, square, first-half of inverse FFT.
  std::vector<Kernel *> headKerns {fftP.get(), transpose.get(), squareConv.get(), transposeT.get()};
    
  // sequence of: second-half of inverse FFT, inverse weighting, carry propagation.
  std::vector<Kernel *> tailKerns {fftW.get(), carryA.get(), carryB.get()};

  // equivalent to sequence of: tailKerns, headKerns.
  std::vector<Kernel *> coreKerns;

  float bitsPerWord = E / (float) N;
  if (bitsPerWord > 18.6f) { log("Warning: high word size of %.2f bits may result in errors\n", bitsPerWord); }
  bool lowBits = bitsPerWord < 13;
  if (lowBits && !args.useLegacy) { log("Note: low word size of %.2f bits forces use of legacy kernels\n", bitsPerWord); }
  if (args.useLegacy || lowBits) {
    coreKerns = tailKerns;
    coreKerns.insert(coreKerns.end(), headKerns.begin(), headKerns.end());
  } else {
    coreKerns = {carryConv.get(), transpose.get(), squareConv.get(), transposeT.get()};
  }

  // The IBDWT convolution squaring loop with carry propagation, on 'data', done nIters times.
  // Optional multiply-by-3 at the end.
  auto modSqLoop = [&](cl_queue q, cl_mem data, int nIters, bool doMul3) {
    assert(nIters > 0);
            
    fftP->setArg(0, data);
    run(headKerns, q);

    carryA->setArg(2, data);
    carryB->setArg(0, data);

    for (int i = 0; i < nIters - 1; ++i) { run(coreKerns, q); }
    
    if (doMul3) {
      carryM->setArg(2, data);
      run({fftW.get(), carryM.get(), carryB.get()}, q);
    } else {
      run(tailKerns, q);
    }

    if (args.timeKernels) { logTimeKernels(coreKerns, nIters); }
  };

  // The modular multiplication a = a * b. Output in 'a'.
  auto modMul = [&](cl_queue q, cl_mem a, cl_mem b) {
    fftP->setArg(0, b);
    transpose->setArg(1, buf3);
    fftH->setArg(0, buf3);
    run(directFftKerns, q);

    fftP->setArg(0, a);
    transpose->setArg(1, buf2);
    fftH->setArg(0, buf2);
    run(directFftKerns, q);

    run({mul.get(), fftH.get(), transposeT.get()}, q);

    carryA->setArg(2, a);
    carryB->setArg(0, a);
    run(tailKerns, q);
  };

  bool isPrime;
  u64 residue;
  int nErrors = 0;
  if (!checkPrime(W, H, E, queue, context, args, &isPrime, &residue, &nErrors, std::move(modSqLoop), std::move(modMul))) {
    return false;
  }
  
  if (!(writeResult(E, isPrime, residue, AID, args.user, args.cpu, nErrors) && worktodoDelete(E))) { return false; }

  return true;
}

int main(int argc, char **argv) {
  initLog();
  
  log("gpuOwL v" VERSION " GPU Mersenne primality checker\n");

  Args args;
  
  if (!args.parse(argc, argv)) { return -1; }

  cl_device_id device = getDevice(args);  
  if (!device) { return -1; }

  if (args.cpu.empty()) { args.cpu = getDeviceName(device); }

  std::string info = getDeviceInfo(device);
  log("%s\n", info.c_str());
  
  Context contextHolder{createContext(device)};
  cl_context context = contextHolder.get();
  Queue queueHolder{makeQueue(device, context)};
  cl_queue queue = queueHolder.get();

  int MAX_2M = 40000000, MAX_4M = 78000000;
  
  
  while (true) {
    char AID[64];
    int E = worktodoReadExponent(AID);
    if (E <= 0) { break; }
    
    int W, H;
    if (Checkpoint::readSize(E, &W, &H)) {
      log("Setting FFT size to %dM based on savefile.\n", W * H * 2 / (1024 * 1024));
    } else {
      int sizeM = args.fftSize ? args.fftSize / (1024 * 1024) : E < MAX_2M ? 2 : E < MAX_4M ? 4 : 8;        
      switch (sizeM) {
      case 2:
        W = H = 1024;
        break;
      case 4:
        W = 1024;
        H = 2048;
        break;
      case 8:
        W = 2048;
        H = 2048;
        break;
      default:
        assert(false);
      }
    }
    
    if (!doIt(device, context, queue, args, AID, E, W, H)) { break; }
  }

  log("\nBye\n");
  for (FILE *f : logFiles) { fclose(f); }
}


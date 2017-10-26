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

#define VERSION "1.6"
#define PROGRAM "gpuowl"

const unsigned BUF_CONST = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
const unsigned BUF_RW    = CL_MEM_READ_WRITE; // | CL_MEM_HOST_NO_ACCESS;

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

static_assert(sizeof(Buffer) == sizeof(cl_mem), "size Buffer");

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
  int itemsPerThread;
  std::string name;

public:
  BaseKernel(cl_program program, int N, const std::string &name, int itemsPerThread) :
    kernel(makeKernel(program, name.c_str())),
    N(N),
    itemsPerThread(itemsPerThread),
    name(name)
  {
    assert(N % itemsPerThread == 0);
  }

  virtual void run(cl_queue q) { ::run(q, kernel.get(), N / itemsPerThread, name); }

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

u32 extra(unsigned N, unsigned E, unsigned k) {
  assert(E & (N - 1));
  u32 step = N - (E & (N - 1));
  return (k * step) & (N - 1);
}

bool isBigWord(unsigned N, unsigned E, unsigned k) {
  u32 step = N - (E & (N - 1));
  return extra(N, E, k) + step < N;
}

u32 bitlen(int N, int E, int k) { return E / N + isBigWord(N, E, k); }

// Sets the weighting vectors direct A and inverse iA (as per IBDWT).
// NTT doesn't use weight vectors.
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
using uint2   = Pair<uint>;

uint2 U2(u32 x, u32 y) { return uint2{x, y}; }

#include "nttshared.h"

// power: a^k
uint2 pow(uint2 a, u32 k) {  
  uint2 x = U2(1, 0);
  for (int i = std::log2(k); i >= 0; --i) {
    x = sq(x);
    if (k & (1 << i)) { x = mul(x, a); }    
  }
  return x;
}

// a^(2^k)
uint2 pow2(uint2 a, u32 k) {
  for (u32 i = 0; i < k; ++i) { a = sq(a); }
  return a;
}

// Returns the primitive root of unity of order N, to the power k.
template<typename T2> T2 root1(uint N, uint k);

template<> float2 root1<float2>(uint N, uint k) {
  double angle = - double(TAU) / N * k;
  float x = cos(angle);
  float y = sin(angle);
  return float2{x, y};
}

template<> double2 root1<double2>(uint N, uint k) {
  long double angle = - TAU / N * k;
  double x = cosl(angle);
  double y = sinl(angle);
  return double2{x, y};
}

// ROOT1_32 ^ 31 == -1, aka "primitive root of unity of order 32" in GF(M(31)^2).
// See "Matters computational", http://www.jjj.de/fxt/fxtbook.pdf .
const uint2 ROOT1_32{1 << 16, 0x4b94532f};

template<> uint2 root1<uint2>(uint N, uint k) {
  uint2 w = pow2(ROOT1_32, 32 - std::log2(N));
  return pow(w, k);
}

template<typename T2>
T2 *trig(T2 *p, int n, int B) {
  for (int i = 0; i < n; ++i) { *p++ = root1<T2>(B, i); }
  return p;
}

// The generated trig table has three regions:
// - a 4096 "full trig table" (a full circle).
// - a region of granularity TAU / (W * H), used in transpose.
// - a region of granularity TAU / (2 * W * H), used in squaring.
template<typename T2>
cl_mem genBigTrig(cl_context context, int W, int H) {
  assert((W == 1024 || W == 2048) && (H == 1024 || H == 2048));
  const int size = H * 2 + W / 2 + W;
  T2 *tab = new T2[size];
  T2 *end = tab;
  end = trig(end, H * 2, H * 2);
  end = trig(end, W / 2, W * H);
  end = trig(end, W,     W * H * 2);
  assert(end - tab == size);
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(T2) * size, tab);
  delete[] tab;
  return buf;
}

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

template<typename T2>
cl_mem genSmallTrig2K(cl_context context) {
  int size = 4 * 512;
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

std::vector<std::unique_ptr<FILE>> logFiles;

void initLog() {
  logFiles.push_back(std::unique_ptr<FILE>(stdout));
  if (auto fo = open("gpuowl.log", "a")) {
#ifdef _DEFAULT_SOURCE
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
  // "residue-checksum":"%08x", 
  
  log("%s\n", buf);
  if (auto fo = open("results.txt", "a")) {
    fprintf(fo.get(), "%s\n", buf);
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
  log("PRP-3 FFT %dK (%d*%d*2) of %d (%.2f bits/word) iteration %d\n", N / 1024, W, H, E, E / float(N), k);
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

string valueDefine(const string &key, uint value) {
  return key + "=" + std::to_string(value) + "u";
}

bool doIt(cl_device_id device, cl_context context, cl_queue queue, const Args &args, const string &AID, int E, int W, int H) {
  assert(W == 1024 || W == 2048);
  assert(H == 1024 || H == 2048);
  assert(W <= H);
  int N = 2 * W * H;
  int nW = W / 256, nH = H / 256;

  string dumpConfig = args.fftKind == Args::FFT_NTT ? "NTT" : args.fftKind == Args::FFT_DP ? "DP" : "SP";
  dumpConfig += string("_") + std::to_string(N / 1024 / 1024) + "M";
  
  std::vector<string> defines {valueDefine("EXP", E), valueDefine("WIDTH", W), valueDefine("HEIGHT", H)};

  switch (args.fftKind) {
  case Args::FFT_NTT: {
    append(defines, "FFT_NTT");
    
    // (2^LOG_ROOT2)^N == 2 (mod M31), so LOG_ROOT2 * N == 1 (mod 31) == 32 (mod 31), so LOG_ROOT2 = 32 / (N % 31) (mod 31).
    
    u32 LOG_ROOT2 = (32 / (N % 31)) % 31;
    append(defines, valueDefine("LOG_ROOT2", LOG_ROOT2));

    // N * 2^(31 - LOG_NWORDS) == 1 (mod M31), used for the 1/N weighting of the iFFT.
    append(defines, valueDefine("LOG_NWORDS", std::log2(N)));    
  }
    break;
    
  case Args::FFT_DP:
    append(defines, "FFT_FP");
    append(defines, "FP_DP");
    break;

  case Args::FFT_SP:
    append(defines, "FFT_FP");
    append(defines, "FP_SP");
    break;

  default:
    assert(false);
  }

  std::unique_ptr<Kernel> fftP, fftW, fftH, carryA, carryM, carryB, transposeW, transposeH, square, multiply, tail, carryConv;

  {
    string clArgs = args.clArgs;
    if (!args.dump.empty()) { clArgs += " -save-temps=" + args.dump + dumpConfig; }
    
    Holder<cl_program> program(compile(device, context, "gpuowl.cl", clArgs, defines));
    if (!program) { return false; }

    auto load = [&program, &args](const string &name, int nWords, int wordsPerThread) {
      Kernel *kernel = new BaseKernel(program.get(), nWords, name, wordsPerThread);
      return args.timeKernels ? new TimedKernel(kernel) : kernel;
    };
    
#define LOAD(name, nWords, wordsPerThread) name.reset(load(#name, nWords, wordsPerThread));
    LOAD(fftP, N, nW * 2);
    LOAD(fftW, N, nW * 2);
    LOAD(fftH, N, nH * 2);
    
    LOAD(carryA, N, 32);
    LOAD(carryM, N, 32);
    LOAD(carryB, N, 32);

    LOAD(transposeW, N, 32);
    LOAD(transposeH, N, 32);
    
    LOAD(square,   N, 4);
    LOAD(multiply, N, 4);
    LOAD(tail,     N, nH * 4);
    
    LOAD(carryConv, N + W * 2, nW * 2);
#undef LOAD
  }

  Buffer bufTrig1K, bufTrig2K, bufBigTrig, bufA, bufI;

  switch (args.fftKind) {
  case Args::FFT_NTT:
    bufTrig1K.reset(genSmallTrig1K<uint2>(context));
    bufTrig2K.reset(genSmallTrig2K<uint2>(context));
    bufBigTrig.reset(genBigTrig<uint2>(context, W, H));
    break;

  case Args::FFT_DP:
    bufTrig1K.reset(genSmallTrig1K<double2>(context));
    bufTrig2K.reset(genSmallTrig2K<double2>(context));
    bufBigTrig.reset(genBigTrig<double2>(context, W, H));
    setupWeights<double>(context, bufA, bufI, W, H, E);
    break;

  case Args::FFT_SP:
    bufTrig1K.reset(genSmallTrig1K<float2>(context));
    bufTrig2K.reset(genSmallTrig2K<float2>(context));
    bufBigTrig.reset(genBigTrig<float2>(context, W, H));
    setupWeights<float>(context, bufA, bufI, W, H, E);
    break;

  default:
    assert(false);
  }

  uint wordSize =
    args.fftKind == Args::FFT_NTT ? sizeof(uint) :
    args.fftKind == Args::FFT_DP  ? sizeof(double) :
    sizeof(float);
  
  uint bufSize = N * wordSize;
  Buffer buf1{makeBuf(    context, BUF_RW, bufSize)};
  Buffer buf2{makeBuf(    context, BUF_RW, bufSize)};
  Buffer buf3{makeBuf(    context, BUF_RW, bufSize)};
  Buffer bufCarry{makeBuf(context, BUF_RW, bufSize)}; // could be N/2 as well.

  int *zero = new int[H + 1]();
  Buffer bufReady{makeBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * (H + 1), zero)};
  delete[] zero;

  Buffer dummy;
  
  cl_mem trigW = (W == 1024) ? bufTrig1K.get() : bufTrig2K.get();
  cl_mem trigH = (H == 1024) ? bufTrig1K.get() : bufTrig2K.get();
  
  fftP->setArgs(dummy, buf1, bufA, trigW);
  fftW->setArgs(buf1, trigW);
  fftH->setArgs(buf2, trigH);
  
  transposeW->setArgs(buf1, buf2, bufBigTrig);
  transposeH->setArgs(buf2, buf1, bufBigTrig);
  
  carryA->setArgs(buf1, bufI, dummy, bufCarry);
  carryM->setArgs(buf1, bufI, dummy, bufCarry);
  carryB->setArgs(dummy, bufCarry);

  square->setArgs(buf2, bufBigTrig);
  multiply->setArgs(buf2, buf3, bufBigTrig);

  tail->setArgs(buf2, trigH, bufBigTrig);
  carryConv->setArgs(buf1, bufCarry, bufReady, bufA, bufI, trigW);

  // the weighting + direct FFT only, stops before square/mul.
  std::vector<Kernel *> directFftKerns {fftP.get(), transposeW.get(), fftH.get()};

  // sequence of: direct FFT, square, first-half of inverse FFT.
  std::vector<Kernel *> headKerns {fftP.get(), transposeW.get(), tail.get(), transposeH.get()};
    
  // sequence of: second-half of inverse FFT, inverse weighting, carry propagation.
  std::vector<Kernel *> tailKerns {fftW.get(), carryA.get(), carryB.get()};

  // equivalent to sequence of: tailKerns, headKerns.
  std::vector<Kernel *> coreKerns {carryConv.get(), transposeW.get(), tail.get(), transposeH.get()};

  float bitsPerWord = E / (float) N;
  if (bitsPerWord > 18.6f) { log("Warning: high word size of %.2f bits may result in errors\n", bitsPerWord); }
  bool lowBits = bitsPerWord < 13;
  if (lowBits && !args.useLegacy) { log("Note: low word size of %.2f bits forces use of legacy kernels\n", bitsPerWord); }
  
  if (args.useLegacy || lowBits) {
    coreKerns = tailKerns;
    coreKerns.insert(coreKerns.end(), headKerns.begin(), headKerns.end());
  }

  if (args.debug) {    
    uint2 *data = new uint2[N / 2]();
    data[0] = U2(3, 0);
    cl_mem bufData = makeBuf(context, CL_MEM_READ_WRITE, sizeof(int) * N);
    fftP->setArg(0, bufData);
    carryA->setArg(2, bufData);
    carryB->setArg(0, bufData);
    write(queue, false, bufData, sizeof(u32) * N, data);
    // write(queue, false, buf1.get(), sizeof(u32) * N, data);

    for (int i = 0; i < 32; ++i) {
      run({fftP.get(), transposeW.get(), fftH.get(), square.get(), fftH.get(), transposeH.get()}, queue);
      run(tailKerns, queue);
      read(queue, true, bufData, sizeof(u32) * N, data);
      printf("%2d: %016llx\n", i + 1, residue(W, H, E, (int *) data));
      // for (int i = 0; i < 64; ++i) { printf("%2d %08x %08x (%d, %d)\n", i, data[i * W].x, data[i * W].y, bitlen(N, E, i * 2), bitlen(N, E, i * 2 + 1)); }
    }
    
    delete[] data;
    return false;
  }
  
  // The IBDWT convolution squaring loop with carry propagation, on 'data', done nIters times.
  // Optional multiply-by-3 at the end.
  auto modSqLoop = [&](cl_queue q, cl_mem data, int nIters, bool doMul3) {
    assert(nIters > 0);
            
    fftP->setArg(0, data);
    run(headKerns, q);

    carryA->setArg(2, data);
    carryB->setArg(0, data);

    for (int i = 0; i < nIters - 1; ++i) {
      // if (((i + 1) & 127) == 0) { finish(q); }
      run(coreKerns, q);
    }
    
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
    transposeW->setArg(1, buf3);
    fftH->setArg(0, buf3);
    run(directFftKerns, q);

    fftP->setArg(0, a);
    transposeW->setArg(1, buf2);
    fftH->setArg(0, buf2);
    run(directFftKerns, q);

    run({multiply.get(), fftH.get(), transposeH.get()}, q);

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
  
  if (!(writeResult(E, isPrime, residue, AID, args.user, args.cpu, nErrors, N) && worktodoDelete(E))) { return false; }

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
}


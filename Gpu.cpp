// Copyright 2017 Mihai Preda.

#include "Gpu.h"

#include "checkpoint.h"
#include "Stats.h"
#include "state.h"
#include "timeutil.h"
#include "args.h"
#include "GCD.h"
#include "Primes.h"
#include "Result.h"
#include "Signal.h"

#include <cmath>
#include <cassert>
#include <algorithm>
#include <gmp.h>

// The assert on sizeof(long) was needed when using mpz_ui with u64, which is not done anymore now.
// static_assert(sizeof(long) == 8, "size long");

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#define TAU (2 * M_PIl)

using double2 = pair<double, double>;

static_assert(sizeof(double2) == 16, "size double2");

// Returns the primitive root of unity of order N, to the power k.
static double2 root1(u32 N, u32 k) {
  long double angle = - TAU / N * k;
  return double2{double(cosl(angle)), double(sinl(angle))};
}

static double2 *smallTrigBlock(int W, int H, double2 *p) {
  for (int line = 1; line < H; ++line) {
    for (int col = 0; col < W; ++col) {
      *p++ = root1(W * H, line * col);
    }
  }
  return p;
}

static cl_mem genSmallTrig(cl_context context, int size, int radix) {
  auto *tab = new double2[size]();
  auto *p = tab + radix;
  int w = 0;
  for (w = radix; w < size; w *= radix) { p = smallTrigBlock(w, std::min(radix, size / w), p); }
  assert(p - tab == size);
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double2) * size, tab);
  delete[] tab;
  return buf;
}

static void setupWeights(cl_context context, Buffer &bufA, Buffer &bufI, int W, int H, int E) {
  int N = 2 * W * H;
  auto weights = genWeights(E, W, H);
  bufA.reset(makeBuf(context, BUF_CONST, sizeof(double) * N, weights.first.data()));
  bufI.reset(makeBuf(context, BUF_CONST, sizeof(double) * N, weights.second.data()));
}

Gpu::~Gpu() {}

Gpu::Gpu(u32 E, u32 W, u32 BIG_H, u32 SMALL_H, int nW, int nH,
         cl_program program, cl_device_id device, cl_context context,
         bool timeKernels, bool useLongCarry) :
  E(E),
  N(W * BIG_H * 2),
  hN(N / 2),
  nW(nW),
  nH(nH),
  bufSize(N * sizeof(double)),
  useLongCarry(useLongCarry),
  useMiddle(BIG_H != SMALL_H),
  gcd(make_unique<GCD>()),
  queue(makeQueue(device, context)),    

#define LOAD(name, workGroups) name(program, queue.get(), device, workGroups, #name, timeKernels)
  LOAD(carryFused, BIG_H + 1),
  LOAD(carryFusedMul, BIG_H + 1),
  LOAD(fftP, BIG_H),
  LOAD(fftW, BIG_H),
  LOAD(fftH, (hN / SMALL_H)),
  LOAD(fftMiddleIn,  hN / (256 * (BIG_H / SMALL_H))),
  LOAD(fftMiddleOut, hN / (256 * (BIG_H / SMALL_H))),
  LOAD(carryA,   nW * (BIG_H/16)),
  LOAD(carryM,   nW * (BIG_H/16)),
  LOAD(carryB,   nW * (BIG_H/16)),
  LOAD(transposeW,   (W/64) * (BIG_H/64)),
  LOAD(transposeH,   (W/64) * (BIG_H/64)),
  LOAD(transposeIn,  (W/64) * (BIG_H/64)),
  LOAD(transposeOut, (W/64) * (BIG_H/64)),
  LOAD(square,   hN / SMALL_H),
  LOAD(multiply, hN / SMALL_H),
  LOAD(multiplySub, hN / SMALL_H),
  LOAD(tailFused, (hN / SMALL_H) / 2),
  LOAD(readResidue, 1),
  LOAD(isNotZero, 256),
  LOAD(isEqual, 256),
#undef LOAD

  bufData( makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
  bufCheck(makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
  bufAux(  makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
  bufBase( makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
  bufAcc(  makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
  
  bufTrigW(genSmallTrig(context, W, nW)),
  bufTrigH(genSmallTrig(context, SMALL_H, nH)),
  buf1{makeBuf(    context, BUF_RW, bufSize)},
  buf2{makeBuf(    context, BUF_RW, bufSize)},
  buf3{makeBuf(    context, BUF_RW, bufSize)},
  bufCarry{makeBuf(context, BUF_RW, bufSize / 2)},
  bufReady{makeBuf(context, BUF_RW, BIG_H * sizeof(int))},
  bufSmallOut(makeBuf(context, CL_MEM_READ_WRITE, 256 * sizeof(int))),
  bufBaseDown(makeBuf(context, BUF_RW, bufSize))
{    
  setupWeights(context, bufA, bufI, W, BIG_H, E);

  carryFused.setFixedArgs(3, bufA, bufI, bufTrigW);
  carryFusedMul.setFixedArgs(3, bufA, bufI, bufTrigW);
    
  fftP.setFixedArgs(2, bufA, bufTrigW);
  fftW.setFixedArgs(1, bufTrigW);
  fftH.setFixedArgs(1, bufTrigH);
    
  carryA.setFixedArgs(3, bufI);
  carryM.setFixedArgs(3, bufI);
  tailFused.setFixedArgs(1, bufTrigH);
    
  queue.zero(bufReady, BIG_H * sizeof(int));
  queue.zero(bufAcc,   N * sizeof(int));
  queue.write(bufAcc, vector<u32>{1});
}

void logTimeKernels(std::initializer_list<Kernel *> kerns) {
  struct Info {
    std::string name;
    StatsInfo stats;
  };

  double total = 0;
  vector<Info> infos;
  for (Kernel *k : kerns) {
    Info info{k->getName(), k->resetStats()};
    infos.push_back(info);
    total += info.stats.sum;
  }

  // std::sort(infos.begin(), infos.end(), [](const Info &a, const Info &b) { return a.stats.sum >= b.stats.sum; });

  for (Info info : infos) {
    StatsInfo stats = info.stats;
    float percent = 100 / total * stats.sum;
    if (true || percent >= .1f) {
      log("%4.1f%% %-14s : %6.0f [%5.0f, %6.0f] us/call   x %5d calls\n",
          percent, info.name.c_str(), stats.mean, stats.low, stats.high, stats.n);
    }
  }
  log("\n");
}

struct FftConfig {
  u32 width, height, middle;
  int fftSize;
  u32 maxExp;

  FftConfig(u32 width, u32 height, u32 middle) :
    width(width),
    height(height),
    middle(middle),
    fftSize(width * height * middle * 2),
    // 17.88 + 0.36 * (24 - log2(n)); Update after feedback on 86700001, FFT 4608 (18.37b/w) being insufficient.
    maxExp(fftSize * (17.77 + 0.33 * (24 - log2(fftSize)))) {
    assert(width  == 512 || width == 1024 || width == 2048 || width == 4096);
    assert(height == 512 || height == 1024 || height == 2048);
    assert(middle == 1 || middle == 5 || middle == 9);
  }
};

static vector<FftConfig> genConfigs() {
  vector<FftConfig> configs;
  for (u32 width : {512, 1024, 2048, 4096}) {
    for (u32 height : {512, 1024, 2048}) {
      for (u32 middle : {1, 5, 9}) {
        configs.push_back(FftConfig(width, height, middle));
      }
    }
  }
  std::sort(configs.begin(), configs.end(), [](const FftConfig &a, const FftConfig &b) {
      if (a.fftSize != b.fftSize) { return (a.fftSize < b.fftSize); }
      assert(a.width != b.width);
      if (a.width == 1024) { return true; }
      if (b.width == 1024) { return false; }
      return (a.width < b.width);
    });
  return configs;
}

static FftConfig getFftConfig(const vector<FftConfig> &configs, u32 E, int argsFftSize) {
  int i = 0;
  int n = int(configs.size());
  // log("A %d %d %d\n", n, argsFftSize, E);
  if (argsFftSize < 10) { // fft delta or not specified.
    while (i < n - 1 && configs[i].maxExp < E) { ++i; }      
    i = max(0, min(i + argsFftSize, n - 1));
  } else { // user-specified fft size.
    while (i < n - 1 && argsFftSize > configs[i].fftSize) { ++i; }      
  }
  return configs[i];
}

vector<int> Gpu::readSmall(Buffer &buf, u32 start) {
  readResidue(buf, bufSmallOut, start);
  return queue.read<int>(bufSmallOut, 128);                    
}

unique_ptr<Gpu> Gpu::make(u32 E, const Args &args) {
  vector<FftConfig> configs = genConfigs();
  if (args.listFFT) {
    log("   FFT  maxExp %4s %4s M\n", "W", "H");
    for (auto c : configs) {
      log("%5.1fM %6.1fM %4d %4d %d\n", c.fftSize / float(1024 * 1024), c.maxExp / float(1000 * 1000), c.width, c.height, c.middle);
    }
    log("\n");
  }
        
  FftConfig config = getFftConfig(configs, E, args.fftSize);
  int WIDTH        = config.width;
  int SMALL_HEIGHT = config.height;
  int MIDDLE       = config.middle;
  int N = WIDTH * SMALL_HEIGHT * MIDDLE * 2;

  string configName = (N % (1024 * 1024)) ? std::to_string(N / 1024) + "K" : std::to_string(N / (1024 * 1024)) + "M";

  int nW = (WIDTH == 1024) ? 4 : 8;
  int nH = (SMALL_HEIGHT == 1024) ? 4 : 8;

  float bitsPerWord = E / float(N);
  string strMiddle = (MIDDLE == 1) ? "" : (string(", Middle ") + std::to_string(MIDDLE));
  log("FFT %dK: Width %d (%dx%d), Height %d (%dx%d)%s; %.2f bits/word\n",
      N / 1024, WIDTH, WIDTH / nW, nW, SMALL_HEIGHT, SMALL_HEIGHT / nH, nH, strMiddle.c_str(), bitsPerWord);

  if (bitsPerWord > 20) {
    log("FFT size too small for exponent (%.2f bits/word).\n", bitsPerWord);
    throw "FFT size too small";
  }
    
  bool useLongCarry = (bitsPerWord < 14.5f)
    || (args.carry == Args::CARRY_LONG)
    || (args.carry == Args::CARRY_AUTO && WIDTH >= 2048);
  
  log("Note: using %s carry kernels\n", useLongCarry ? "long" : "short");

  string clArgs = args.clArgs;
  if (!args.dump.empty()) { clArgs += " -save-temps=" + args.dump + "/" + configName; }

  bool timeKernels = args.timeKernels;
    
  cl_device_id device = getDevice(args.device);
  if (!device) { throw "No OpenCL device"; }

  log("%s\n", getLongInfo(device).c_str());
  // if (args.cpu.empty()) { args.cpu = getShortInfo(device); }

  Context context(createContext(device));
  Holder<cl_program> program(compile(device, context.get(), "gpuowl", clArgs,
                                     {{"EXP", E}, {"WIDTH", WIDTH}, {"SMALL_HEIGHT", SMALL_HEIGHT}, {"MIDDLE", MIDDLE}},
                                     args.usePrecompiled));
  if (!program) { throw "OpenCL compilation"; }

  return make_unique<Gpu>(E, WIDTH, SMALL_HEIGHT * MIDDLE, SMALL_HEIGHT, nW, nH,
                          program.get(), device, context.get(), timeKernels, useLongCarry);
}

vector<u32> Gpu::readData()  { return compactBits(readOut(bufData),  E); }
vector<u32> Gpu::readCheck() { return compactBits(readOut(bufCheck), E); }
vector<u32> Gpu::readAcc()   { return compactBits(readOut(bufAcc), E); }

vector<u32> Gpu::writeData(const vector<u32> &v) {
  writeIn(expandBits(v, N, E), bufData);
  return v;
}

vector<u32> Gpu::writeCheck(const vector<u32> &v) {
  writeIn(expandBits(v, N, E), bufCheck);
  return v;
}

void Gpu::writeState(const vector<u32> &check, const vector<u32> &base, u32 blockSize) {
  assert(blockSize > 0);
    
  writeCheck(check);
  queue.copy<int>(bufCheck, bufData, N);
  queue.copy<int>(bufCheck, bufBase, N);

  u32 n = 0;
  for (n = 1; blockSize % (2 * n) == 0; n *= 2) {
    dataLoopMul(vector<bool>(n));
    modMul(bufBase, bufData);
    queue.copy<int>(bufData, bufBase, N);
  }

  assert((n & (n - 1)) == 0);
  assert(blockSize % n == 0);
    
  blockSize /= n;
  for (u32 i = 0; i < blockSize - 1; ++i) {
    dataLoopMul(vector<bool>(n));
    modMul(bufBase, bufData);
  }
    
  writeBase(base);
  modMul(bufBase, bufData);
}

void Gpu::updateCheck() { modMul(bufData, bufCheck); }
  
bool Gpu::doCheck(int blockSize) {
  queue.copy<int>(bufCheck, bufAux, N);
  modSqLoopMul(bufAux, vector<bool>(blockSize));
  modMul(bufBase, bufAux);
  updateCheck();
  return equalNotZero(bufCheck, bufAux);
}

u32 Gpu::dataLoopAcc(u32 kBegin, u32 kEnd, const unordered_set<u32> &kset) {
  assert(kEnd > kBegin);
  vector<bool> accs;
  u32 nAcc = 0;
  for (u32 k = kBegin; k < kEnd; ++k) {
    bool on = kset.count(k);
    accs.push_back(on);
    nAcc += on;
  }
  assert(accs.size() == kEnd - kBegin);
  modSqLoopAcc(bufData, accs);
  return nAcc;
}

void Gpu::dataLoopMul(const vector<bool> &muls) { modSqLoopMul(bufData, muls); }

u64 Gpu::dataResidue() { return bufResidue(bufData); }
u64 Gpu::checkResidue() { return bufResidue(bufCheck); }

void Gpu::logTimeKernels() {
  ::logTimeKernels({&carryFused, &carryFusedMul, &fftP, &fftW, &fftH, &fftMiddleIn, &fftMiddleOut,
        &carryA, &carryM, &carryB,
        &transposeW, &transposeH, &transposeIn, &transposeOut,
        &square, &multiply, &multiplySub, &tailFused, &readResidue, &isNotZero, &isEqual});
}

void Gpu::tW(Buffer &in, Buffer &out) {
  transposeW(in, out);
  if (useMiddle) { fftMiddleIn(out); }
}

void Gpu::tH(Buffer &in, Buffer &out) {
  if (useMiddle) { fftMiddleOut(in); }
  transposeH(in, out);
}

vector<u32> Gpu::writeBase(const vector<u32> &v) {
  writeIn(expandBits(v, N, E), bufBase);
  fftP(bufBase, buf1);
  tW(buf1, bufBaseDown);
  fftH(bufBaseDown);
  return v;
}
  
vector<int> Gpu::readOut(Buffer &buf) {
  transposeOut(buf, bufAux);
  return queue.read<int>(bufAux, N);
}
  
void Gpu::writeIn(const vector<int> &words, Buffer &buf) {
  queue.write(bufAux, words);
  transposeIn(bufAux, buf);
}

void Gpu::modSqLoopMul(Buffer &io, const vector<bool> &muls) {
  assert(!muls.empty());
  bool dataIsOut = true;
        
  for (auto it = muls.begin(), end = muls.end(); it < end; ++it) {
    if (dataIsOut) { fftP(io, buf1); }
    tW(buf1, buf2);
    tailFused(buf2);
    tH(buf2, buf1);

    dataIsOut = useLongCarry || it == prev(end);
    if (dataIsOut) {
      fftW(buf1);
      *it ? carryM(buf1, io, bufCarry) : carryA(buf1, io, bufCarry);
      carryB(io, bufCarry);
    } else {
      *it ? carryFusedMul(buf1, bufCarry, bufReady) : carryFused(buf1, bufCarry, bufReady);
    }
  }
}

void Gpu::exitKerns(Buffer &buf, Buffer &bufWords) {
  fftW(buf);
  carryA(buf, bufWords, bufCarry);
  carryB(bufWords, bufCarry);
}
  
void Gpu::modSqLoopAcc(Buffer &io, const vector<bool> &accs) {
  assert(!accs.empty());
  bool dataIsOut = true;
  bool accIsOut  = true;
    
  for (auto it = accs.begin(), end = accs.end(); it < end; ++it) {
    if (dataIsOut) { fftP(io, buf1); }
    tW(buf1, buf2);
    
    if (*it) {
      fftH(buf2);
      if (accIsOut) { fftP(bufAcc, buf3); }
      tW(buf3, buf1);
      fftH(buf1);
      multiplySub(buf1, buf2, bufBaseDown);
      fftH(buf1);
      tH(buf1, buf3);
      square(buf2);
      fftH(buf2);

      accIsOut = useLongCarry || !any_of(next(it), end, [](bool on) {return on; });
      if (accIsOut) {
        exitKerns(buf3, bufAcc);
      } else {
        carryFused(buf3, bufCarry, bufReady);
      }
    } else {
      tailFused(buf2);
    }
      
    tH(buf2, buf1);

    dataIsOut = useLongCarry || it == prev(end);
    if (dataIsOut) {
      exitKerns(buf1, io);
    } else {
      carryFused(buf1, bufCarry, bufReady);
    }
  }
}

// The modular multiplication io *= in.
void Gpu::modMul(Buffer &in, Buffer &io) {
  fftP(in, buf1);
  tW(buf1, buf3);
    
  fftP(io, buf1);
  tW(buf1, buf2);
    
  fftH(buf2);
  fftH(buf3);
  multiply(buf2, buf3);
  fftH(buf2);

  tH(buf2, buf1);    

  fftW(buf1);
  carryA(buf1, io, bufCarry);
  carryB(io, bufCarry);
};

bool Gpu::equalNotZero(Buffer &buf1, Buffer &buf2) {
  queue.zero(bufSmallOut, sizeof(int));
  u32 sizeBytes = N * sizeof(int);
  isNotZero(sizeBytes, buf1, bufSmallOut);
  isEqual(sizeBytes, buf1, buf2, bufSmallOut);
  return queue.read<int>(bufSmallOut, 1)[0];
}
  
u64 Gpu::bufResidue(Buffer &buf) {
  u32 earlyStart = N/2 - 32;
  vector<int> readBuf = readSmall(buf, earlyStart);
  return residueFromRaw(E, N, readBuf);
}

static string makeLogStr(int E, int k, u64 res, const StatsInfo &info, u32 nIters = 0) {
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

static void doLog(int E, int k, long timeCheck, u64 res, bool checkOK, Stats &stats) {
  log("%s %s (check %.2fs)\n",
      checkOK ? "OK" : "EE",
      makeLogStr(E, k, res, stats.getStats()).c_str(),
      timeCheck * .001f);
  stats.reset();
}

static void doSmallLog(int E, int k, u64 res, Stats &stats) {
  log("   %s\n", makeLogStr(E, k, res, stats.getStats()).c_str());
  stats.reset();
}

static void powerSmooth(mpz_t a, u32 exp, u32 B1, u32 B2 = 0) {
  if (B2 == 0) { B2 = B1; }
  assert(B2 >= sqrt(B1));

  mpz_set_ui(a, exp);
  mpz_mul_2exp(a, a, 20); // boost 2s.
  // mpz_set_ui(a, u64(exp) << 20); 

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

// Residue from compacted words.
static u64 residue(const std::vector<u32> &words) { return (u64(words[1]) << 32) | words[0]; }

static vector<u32> bitNeg(const vector<u32> &v) {
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
    this->dataLoopMul(vector<bool>(bits.begin() + k, bits.begin() + (k + nIts)));
    queue.finish();
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

static vector<u32> kselect(u32 E, u32 B1, u32 B2) {
  if (!B1) { return vector<u32>(); }

  Primes primes(B2 + 1);
  vector<bool> covered(E);
  vector<bool> on(E);
  
  // u32 prevP = 0;  
  for (u32 p : primes.from(B1)) {
    u32 z = primes.zn2(p);
    if (z < E) {
      if (!covered[z]) {
        // assert(!on[z]);
        for (u32 d : primes.divisors(z)) {
          covered[d] = true;
          on[d] = false;
        }
        on[z] = true;
      }
    }
  }

  vector<u32> ret;
  for (u32 k = 0; k < E; ++k) {
    if (on[k]) {
      ret.push_back(k);
      // if (k % 6986 == 0) { log("K(6986): %u\n", k); }
    }
  }
  return ret;
}

static auto asSet(const vector<u32> &v) { return unordered_set<u32>(v.begin(), v.end()); }

PRPResult Gpu::isPrimePRP(u32 E, const Args &args, u32 B1, u32 B2) {
  u32 N = this->getFFTSize();
  assert(B2 == 0 || B2 >= B1);
  if (B1 != 0 && B2 == 0) { B2 = E; }
  log("PRP M(%d), FFT %dK, %.2f bits/word, B1 %u, B2 %u\n", E, N/1024, E / float(N), B1, B2);

  future<vector<u32>> ksetFuture;
  if (B1 != 0) { ksetFuture = async(launch::async, kselect, E, B1, B2); }
  
  if (!PRPState::exists(E)) {
    auto[check, base] = seedPRP(E, B1);
    PRPState{0, B1, args.blockSize, residue(base), check, base}.save(E);
  }

  PRPState loaded = loadPRP(this, E, B1, args.blockSize);
  u32 k = loaded.k;
  
  if (loaded.B1 != B1) {
    log("B1 mismatch %u %u\n", B1, loaded.B1);
    throw "B1 mismatch";
  }
  
  u32 blockSize = loaded.blockSize;

  const u32 kEnd = E - 1; // Type-4 per http://www.mersenneforum.org/showpost.php?p=468378&postcount=209
  assert(k < kEnd);

  vector<u32> base = loaded.base;
  
  const u64 baseRes64 = residue(base);
  assert(blockSize > 0 && 10000 % blockSize == 0);
  const u32 checkStep = blockSize * blockSize;
  
  int startK = k;

  unordered_set<u32> kset;
  if (ksetFuture.valid()) {
    log("Please wait for P-1 trial points selection..\n");
    ksetFuture.wait();
    kset = asSet(ksetFuture.get());
  }
  
  log("Selected %u P-1 trial points\n", u32(kset.size()));
  
  Signal signal;
  Stats stats;

  // Number of sequential errors (with no success in between). If this ever gets high enough, stop.
  int nSeqErrors = 0;

  bool isPrime = false;
  Timer timer;

  int nGcdAcc = 0;
  u64 finalRes64 = 0;
  while (true) {
    assert(k % blockSize == 0);
    if (k < kEnd && k + blockSize >= kEnd) {
      nGcdAcc += this->dataLoopAcc(k, kEnd, kset);
      auto words = this->roundtripData();
      finalRes64 = residue(words);
      isPrime = (words == base || words == bitNeg(base));

      log("%s %8d / %d, %016llx (base %016llx)\n", isPrime ? "PP" : "CC", kEnd, E, finalRes64, baseRes64);
      
      int itersLeft = blockSize - (kEnd - k);
      // assert(itersLeft > 0);
      if (itersLeft > 0) { nGcdAcc += this->dataLoopAcc(kEnd, kEnd + itersLeft, kset); }
    } else {
      nGcdAcc += this->dataLoopAcc(k, k + blockSize, kset);
    }
    k += blockSize;

    u64 res64 = this->dataResidue();

    if (gcd->isReady()) {
      string factor = gcd->get();
      if (!factor.empty()) {
        log("GCD: %s\n", factor.c_str());
        return PRPResult{factor, false, 0, baseRes64};
      }
    }
        
    auto delta = timer.deltaMillis();
    stats.add(delta * (1.0f / blockSize));
    bool doStop = signal.stopRequested();
    if (doStop) {
      log("Stopping, please wait..\n");
      signal.release();
    }

    bool doCheck = (k % checkStep == 0) || (k >= kEnd && k < kEnd + blockSize) || doStop || (k - startK == 2 * blockSize);

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

    bool doSave = (k < kEnd || k >= kEnd + blockSize) && ok;
    if (doSave) { PRPState{k, B1, blockSize, res64, check, base}.save(E); }
    
    doLog(E, k, timer.deltaMillis(), res64, ok, stats);
    
    bool wantGCD = ok && (nGcdAcc > 10000 || doStop);
    if (wantGCD) {
      if (gcd->isOngoing()) {
        log("Waiting for GCD to finish..\n");
        gcd->wait();
      }

      if (gcd->isReady()) {
        string factor = gcd->get();
        if (!factor.empty()) {
          log("GCD: %s\n", factor.c_str());
          return PRPResult{factor, false, 0, baseRes64};
        }
      }

      assert(!gcd->isOngoing());
      if (nGcdAcc) {
        log("Starting GCD over %u points\n", nGcdAcc);
        gcd->start(E, this->readAcc(), 0);
        nGcdAcc = 0;
      }
    }

    if (ok) {
      if (isPrime || k >= kEnd) { return PRPResult{"", isPrime, finalRes64, baseRes64}; }
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
      }
      if (gcd->isReady()) {
        string factor = gcd->get();
        if (!factor.empty()) {
          log("GCD: %s\n", factor.c_str());
          return PRPResult{factor, false, 0, baseRes64};
        }
      }
      assert(!gcd->isOngoing());
      assert(nGcdAcc == 0 || !ok);
      throw "stop requested";
    }
  }
}

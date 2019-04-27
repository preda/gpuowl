// Copyright 2017 Mihai Preda.

#include "Gpu.h"

#include "Pm1Plan.h"
#include "checkpoint.h"
#include "state.h"
#include "Args.h"
#include "Signal.h"
#include "FFTConfig.h"
#include "GmpUtil.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <future>

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

static string CL_SOURCE =
#include "gpuowl-wrap.cl"
;

static cl_program compile(const Args& args, cl_context context, u32 E, u32 WIDTH, u32 SMALL_HEIGHT, u32 MIDDLE) {
  string clArgs = args.dump.empty() ? ""s : (" -save-temps="s + args.dump + "/" + numberK(WIDTH * SMALL_HEIGHT * MIDDLE * 2));
  cl_program program = compile({getDevice(args.device)}, context, CL_SOURCE, clArgs,
                 {{"EXP", E}, {"WIDTH", WIDTH}, {"SMALL_HEIGHT", SMALL_HEIGHT}, {"MIDDLE", MIDDLE}});
  if (!program) { throw "OpenCL compilation"; }
  return program;
}

Gpu::Gpu(const Args& args, u32 E, u32 W, u32 BIG_H, u32 SMALL_H, int nW, int nH,
         cl_device_id device, bool timeKernels, bool useLongCarry) :
  E(E),
  N(W * BIG_H * 2),
  hN(N / 2),
  nW(nW),
  nH(nH),
  bufSize(N * sizeof(double)),
  useLongCarry(useLongCarry),
  useMiddle(BIG_H != SMALL_H),
  device(device),
  context(createContext(device)),
  program(compile(args, context.get(), E, W, SMALL_H, BIG_H / SMALL_H)),
  queue(makeQueue(device, context.get())),  

#define LOAD(name, workGroups) name(program.get(), queue.get(), device, workGroups, #name, timeKernels)
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
  LOAD(multiply, hN / SMALL_H),
  LOAD(square, hN/SMALL_H),
  LOAD(tailFused, (hN / SMALL_H) / 2),
  LOAD(tailFusedMulDelta, (hN / SMALL_H) / 2),
  LOAD(readResidue, 1),
  LOAD(isNotZero, 256),
  LOAD(isEqual, 256),
#undef LOAD

  bufTrigW(genSmallTrig(context.get(), W, nW)),
  bufTrigH(genSmallTrig(context.get(), SMALL_H, nH)),
  
  bufData(makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
  bufAux( makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
  
  bufCarry{makeBuf(context, BUF_RW, bufSize / 2)},
  bufReady{makeBuf(context, BUF_RW, BIG_H * sizeof(int))},
  bufSmallOut(makeBuf(context, CL_MEM_READ_WRITE, 256 * sizeof(int)))
{
  // dumpBinary(program.get(), "isa.bin");
  program.reset();
  setupWeights(context.get(), bufWeightA, bufWeightI, W, BIG_H, E);

  carryFused.setFixedArgs(   1, bufCarry, bufReady, bufWeightA, bufWeightI, bufTrigW);
  carryFusedMul.setFixedArgs(1, bufCarry, bufReady, bufWeightA, bufWeightI, bufTrigW);
  fftP.setFixedArgs(2, bufWeightA, bufTrigW);
  fftW.setFixedArgs(1, bufTrigW);
  fftH.setFixedArgs(1, bufTrigH);
    
  carryA.setFixedArgs(2, bufCarry, bufWeightI);
  carryM.setFixedArgs(2, bufCarry, bufWeightI);
  carryB.setFixedArgs(1, bufCarry);
  tailFused.setFixedArgs(1, bufTrigH);
  tailFusedMulDelta.setFixedArgs(3, bufTrigH);
    
  queue.zero(bufReady, BIG_H * sizeof(int));
}

void logTimeKernels(std::initializer_list<Kernel *> kerns) {
  double total = 0;
  vector<pair<TimeInfo, string>> infos;
  for (Kernel *k : kerns) {
    auto s = k->resetStats();
    if (s.total > 0 && s.n > 0) {
      infos.push_back(make_pair(s, k->getName()));
      total += s.total;
    }
  }

  std::sort(infos.begin(), infos.end(), [](const auto& a, const auto& b){ return a.first.total > b.first.total; });

  for (auto& [stats, name]: infos) {
    float percent = 100 / total * stats.total;
    if (percent >= .01f) {
      log("%5.2f%% %-14s : %6.0f us/call x %5d calls\n",
          percent, name.c_str(), stats.total / stats.n, stats.n);
    }
  }
  log("\n");
}

static FFTConfig getFFTConfig(const vector<FFTConfig> &configs, u32 E, int argsFftSize) {
  int i = 0;
  int n = int(configs.size());
  if (argsFftSize < 10) { // fft delta or not specified.
    while (i < n - 1 && configs[i].maxExp < E) { ++i; }      
    i = max(0, min(i + argsFftSize, n - 1));
  } else { // user-specified fft size.
    while (i < n - 1 && u32(argsFftSize) > configs[i].fftSize) { ++i; }      
  }
  return configs[i];
}

vector<int> Gpu::readSmall(Buffer &buf, u32 start) {
  readResidue(buf, bufSmallOut, start);
  return queue.read<int>(bufSmallOut, 128);                    
}

unique_ptr<Gpu> Gpu::make(u32 E, const Args &args) {
  vector<FFTConfig> configs = FFTConfig::genConfigs();
        
  FFTConfig config = getFFTConfig(configs, E, args.fftSize);
  int WIDTH        = config.width;
  int SMALL_HEIGHT = config.height;
  int MIDDLE       = config.middle;
  int N = WIDTH * SMALL_HEIGHT * MIDDLE * 2;

  int nW = (WIDTH == 1024 || WIDTH == 256) ? 4 : 8;
  int nH = (SMALL_HEIGHT == 1024 || SMALL_HEIGHT == 256) ? 4 : 8;

  float bitsPerWord = E / float(N);
  string strMiddle = (MIDDLE == 1) ? "" : (string(", Middle ") + std::to_string(MIDDLE));
  log("%u FFT %dK: Width %dx%d, Height %dx%d%s; %.2f bits/word\n",
      E, N / 1024, WIDTH / nW, nW, SMALL_HEIGHT / nH, nH, strMiddle.c_str(), bitsPerWord);

  if (bitsPerWord > 20) {
    log("FFT size too small for exponent (%.2f bits/word).\n", bitsPerWord);
    throw "FFT size too small";
  }

  if (bitsPerWord < 1.5) {
    log("FFT size too large for exponent (%.2f bits/word).\n", bitsPerWord);
    throw "FFT size too large";
  }
    
  bool useLongCarry = (bitsPerWord < 14.5f)
    || (args.carry == Args::CARRY_LONG)
    || (args.carry == Args::CARRY_AUTO && WIDTH >= 2048);
  
  log("using %s carry kernels\n", useLongCarry ? "long" : "short");

  bool timeKernels = args.timeKernels;
    
  return make_unique<Gpu>(args, E, WIDTH, SMALL_HEIGHT * MIDDLE, SMALL_HEIGHT, nW, nH,
                          getDevice(args.device), timeKernels, useLongCarry);
}

vector<u32> Gpu::readData()  { return compactBits(readOut(bufData),  E); }
vector<u32> Gpu::readCheck() { return compactBits(readOut(bufCheck), E); }

vector<u32> Gpu::writeData(const vector<u32> &v) {
  writeIn(v, bufData);
  return v;
}

vector<u32> Gpu::writeCheck(const vector<u32> &v) {
  writeIn(v, bufCheck);
  return v;
}

// The modular multiplication io *= in.
void Gpu::modMul(Buffer& in, bool mul3, Buffer& buf1, Buffer& buf2, Buffer& buf3, Buffer& io) {
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
  mul3 ? carryM(buf1, io) : carryA(buf1, io);
  carryB(io);
};

void Gpu::writeState(const vector<u32> &check, u32 blockSize, Buffer& buf1, Buffer& buf2, Buffer& buf3) {
  assert(blockSize > 0);
    
  writeCheck(check);
  queue.copy<int>(bufCheck, bufData, N);
  queue.copy<int>(bufCheck, bufAux, N);

  u32 n = 0;
  for (n = 1; blockSize % (2 * n) == 0; n *= 2) {
    modSqLoop(n, false, buf1, buf2, bufData);  // dataLoop(n);
    modMul(bufAux, false, buf1, buf2, buf3, bufData);
    queue.copy<int>(bufData, bufAux, N);
  }

  assert((n & (n - 1)) == 0);
  assert(blockSize % n == 0);
    
  blockSize /= n;
  assert(blockSize >= 2);
  
  for (u32 i = 0; i < blockSize - 2; ++i) {
    modSqLoop(n, false, buf1, buf2, bufData); // dataLoop(n);
    modMul(bufAux, false, buf1, buf2, buf3, bufData);
  }
  
  modSqLoop(n, false, buf1, buf2, bufData);  // dataLoop(n);
  modMul(bufAux, true, buf1, buf2, buf3, bufData);
}

void Gpu::updateCheck(Buffer& buf1, Buffer& buf2, Buffer& buf3) { modMul(bufData, false, buf1, buf2, buf3, bufCheck); }
  
bool Gpu::doCheck(int blockSize, Buffer& buf1, Buffer& buf2, Buffer& buf3) {
  queue.copy<int>(bufCheck, bufAux, N);
  modSqLoop(blockSize, true, buf1, buf2, bufAux);
  updateCheck(buf1, buf2, buf3);
  return equalNotZero(bufCheck, bufAux);
}

void Gpu::logTimeKernels() {
  ::logTimeKernels({&carryFused,
        &fftP, &fftW, &fftH, &fftMiddleIn, &fftMiddleOut,
        &carryA, &carryM, &carryB,
        &transposeW, &transposeH, &transposeIn, &transposeOut,
        &multiply, &square, &tailFused,
        &readResidue, &isNotZero, &isEqual});
}

void Gpu::tW(Buffer &in, Buffer &out) {
  transposeW(in, out);
  if (useMiddle) { fftMiddleIn(out); }
}

void Gpu::tH(Buffer &in, Buffer &out) {
  if (useMiddle) { fftMiddleOut(in); }
  transposeH(in, out);
}
  
vector<int> Gpu::readOut(Buffer &buf) {
  transposeOut(buf, bufAux);
  return queue.read<int>(bufAux, N);
}

void Gpu::writeIn(const vector<u32> &words, Buffer &buf) { writeIn(expandBits(words, N, E), buf); }

void Gpu::writeIn(const vector<int> &words, Buffer &buf) {
  queue.write(bufAux, words);
  transposeIn(bufAux, buf);
}

// io *= in; with buffers in low position.
void Gpu::multiplyLow(Buffer& in, Buffer& tmp, Buffer& io) {
  multiply(io, in);
  fftH(io);
  tH(io, tmp);
  carryFused(tmp);
  tW(tmp, io);
  fftH(io);
}

// Auxiliary performing the top half of the cycle (excluding the buttom tailFused).
void Gpu::topHalf(Buffer& tmp, Buffer& io) {
  tH(io, tmp);
  carryFused(tmp);
  tW(tmp, io);
}

// See "left-to-right binary exponentiation" on wikipedia
// Computes out := base**exp
// All buffers are in "low" position.
void Gpu::exponentiate(Buffer& base, u64 exp, Buffer& tmp, Buffer& out) {
  if (exp == 0) {
    queue.zero(out, N * sizeof(u32));
    u32 data = 1;
    fillBuf(queue.get(), out, &data, sizeof(data));
    // write(queue.get(), false, out, sizeof(u32), &data);
    fftP(out, tmp);
    tW(tmp, out);    
  } else {
    queue.copy<double>(base, out, N);
    if (exp == 1) { return; }

    int p = 63;
    while ((exp >> p) == 0) { --p; }
    assert(p >= 1);

    // square from "low" position.
    square(out);
    fftH(out);
    topHalf(tmp, out);
  
    while (true) {
      --p;
      if ((exp >> p) & 1) {
        fftH(out); // to low
      
        // multiply from low
        multiply(out, base);
        fftH(out);
        topHalf(tmp, out);
      }
      if (p <= 0) { break; }

      // square
      tailFused(out);
      topHalf(tmp, out);
    }
  }

  fftH(out); // to low
}

void Gpu::coreStep(bool leadIn, bool leadOut, bool mul3, Buffer& buf1, Buffer& buf2, Buffer& io) {
  if (leadIn) { fftP(io, buf1); }
  tW(buf1, buf2);
  tailFused(buf2);
  tH(buf2, buf1);

  if (leadOut) {
    fftW(buf1);
    mul3 ? carryM(buf1, io) : carryA(buf1, io);
    carryB(io);
  } else {
    mul3 ? carryFusedMul(buf1) : carryFused(buf1);
  }  
}

void Gpu::modSqLoop(u32 reps, bool mul3, Buffer& buf1, Buffer& buf2, Buffer& io) {
  assert(reps > 0);
  bool leadIn = true;
        
  for (u32 i = 0; i < reps; ++i) {
    bool leadOut = useLongCarry || (i == reps - 1);
    coreStep(leadIn, leadOut, mul3 && (i == reps - 1), buf1, buf2, io);
    leadIn = leadOut;
  }
}

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

static string getETA(u32 step, u32 total, float msPerStep) {
  // assert(step <= total);
  int etaMins = (total - step) * msPerStep * (1 / 60000.f) + .5f;
  int days  = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins  = etaMins % 60;
  char buf[64];
  snprintf(buf, sizeof(buf), "%dd %02d:%02d", days, hours, mins);
  return string(buf);
}

static string makeLogStr(u32 E, string status, int k, u64 res, TimeInfo info, u32 nIters) {
  float msPerSq = info.total / info.n;
  char buf[256];
  string ghzStr;
  
  snprintf(buf, sizeof(buf), "%u %2s %8d %5.2f%%; %.2f ms/sq;%s ETA %s; %016llx",
           E, status.c_str(), k, k / float(nIters) * 100,
           msPerSq, ghzStr.c_str(), getETA(k, nIters, msPerSq).c_str(), res);
  return buf;
}

static void doLog(int E, int k, u32 timeCheck, u64 res, bool checkOK, TimeInfo &stats, u32 nIters) {
  log("%s (check %.2fs)\n",      
      makeLogStr(E, checkOK ? "OK" : "EE", k, res, stats, nIters).c_str(),
      timeCheck * .001f);
  stats.reset();
}

static void doSmallLog(int E, int k, u64 res, TimeInfo &stats, u32 nIters) {
  log("%s\n", makeLogStr(E, "", k, res, stats, nIters).c_str());
  stats.reset();
}

static bool equalMinus3(const vector<u32> &a) {
  if (a[0] != ~3u) { return false; }
  for (auto it = next(a.begin()); it != prev(a.end()); ++it) { if (~*it) { return false; }}
  return true;
}

PRPState Gpu::loadPRP(u32 E, u32 iniBlockSize, Buffer& buf1, Buffer& buf2, Buffer& buf3) {
  auto loaded = PRPState::load(E, iniBlockSize);

  writeState(loaded.check, loaded.blockSize, buf1, buf2, buf3);

  u64 res64 = dataResidue();
  bool ok = (res64 == loaded.res64);
  updateCheck(buf1, buf2, buf3);
  if (!ok) {
#ifdef __MINGW64__
    log("%u EE loaded: %d, blockSize %d, %016I64x (expected %016I64x)\n", E, loaded.k, loaded.blockSize, res64, loaded.res64);
#else
    log("%u EE loaded: %d, blockSize %d, %016llx (expected %016llxx)\n",  E, loaded.k, loaded.blockSize, res64, loaded.res64);
#endif          
    throw "error on load";
  }

  return loaded;
}

pair<bool, u64> Gpu::isPrimePRP(u32 E, const Args &args) {
  Buffer buf1(makeBuf(context, BUF_RW, bufSize));
  Buffer buf2(makeBuf(context, BUF_RW, bufSize));
  Buffer buf3(makeBuf(context, BUF_RW, bufSize));
  
  bufCheck.reset(makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int)));
  
  PRPState loaded = loadPRP(E, args.blockSize, buf1, buf2, buf3);

  u32 k = loaded.k;
  u32 blockSize = loaded.blockSize;
  assert(blockSize > 0 && 10000 % blockSize == 0);
  
  const u32 kEnd = E - 1; // Type-4 per http://www.mersenneforum.org/showpost.php?p=468378&postcount=209
  assert(k < kEnd);

  const u32 checkStep = blockSize * blockSize;
  
  u32 startK = k;
  
  Signal signal;
  TimeInfo stats;

  // Number of sequential errors (with no success in between). If this ever gets high enough, stop.
  int nSeqErrors = 0;

  bool isPrime = false;
  Timer timer;

  u64 finalRes64 = 0;
  u32 nTotalIters = ((kEnd - 1) / blockSize + 1) * blockSize;
  while (true) {
    assert(k % blockSize == 0);
    if (k < kEnd && k + blockSize >= kEnd) {
      modSqLoop(kEnd - k, false, buf1, buf2, bufData);
      auto words = this->roundtripData();
      finalRes64 = residue(words);
      isPrime = equalMinus3(words);
#ifdef __MINGW64__
      log("%s %8d / %d, %016I64x\n", isPrime ? "PP" : "CC", kEnd, E, finalRes64);
#else
      log("%s %8d / %d, %016llx\n", isPrime ? "PP" : "CC", kEnd, E, finalRes64);
#endif

      // Scream if ever [again] we get residue 0xfffffffffffffffc and !isprime, likely indicating a bug and a missed prime.
      assert(finalRes64 != u64(-3) || isPrime);
      
      int itersLeft = blockSize - (kEnd - k);
      if (itersLeft > 0) { modSqLoop(itersLeft, false, buf1, buf2, bufData); }
    } else {
      modSqLoop(blockSize, false, buf1, buf2, bufData);
    }
    k += blockSize;

    queue.finish();
        
    stats.add(timer.deltaMillis(), blockSize);
    bool doStop = signal.stopRequested();
    if (doStop) {
      log("Stopping, please wait..\n");
      signal.release();
    }

    bool doCheck = (k % checkStep == 0) || (k >= kEnd && k < kEnd + blockSize) || doStop || (k - startK == 2 * blockSize);
    
    if (!doCheck) {
      this->updateCheck(buf1, buf2, buf3);
      if (k % args.logStep == 0) {
        doSmallLog(E, k, dataResidue(), stats, nTotalIters);
        if (args.timeKernels) { this->logTimeKernels(); }
      }
      continue;
    }

    vector<u32> check = this->roundtripCheck();
    bool ok = this->doCheck(blockSize, buf1, buf2, buf3);

    u64 res64 = dataResidue();

    // the check time (above) is accounted separately, not added to iteration time.
    doLog(E, k, timer.deltaMillis(), res64, ok, stats, nTotalIters);
    
    if (ok) {
      if (k < kEnd) { PRPState{k, blockSize, res64, check}.save(E); }
      if (isPrime || k >= kEnd) { return {isPrime, finalRes64}; }
      nSeqErrors = 0;      
    } else {
      if (++nSeqErrors > 2) {
        log("%d sequential errors, will stop.\n", nSeqErrors);
        throw "too many errors";
      }
      
      auto loaded = loadPRP(E, blockSize, buf1, buf2, buf3);
      k = loaded.k;
      assert(blockSize == loaded.blockSize);
    }
    if (args.timeKernels) { this->logTimeKernels(); }
    if (doStop) { throw "stop requested"; }
  }
}

bool isRelPrime(u32 D, u32 j);

static u32 getNBuffers(u32 maxBuffers) {
  u32 nBufs = 0;
  for (u32 i = maxBuffers; i >= 24; --i) {
    if (2880u % i == 0) {
      nBufs = i;
      break;
    }
  }
  return nBufs;
}


/*
Buffer allocBuf(cl_device_id device, size_t size) {
  if (getFreeMem(device) < size) { throw gpu_bad_alloc(); }
}
*/

std::variant<string, vector<u32>> Gpu::factorPM1(u32 E, const Args& args, u32 B1, u32 B2) {
  assert(B1 && B2 && B2 > B1);
  
  vector<bool> bits = powerSmoothBitsRev(E, B1);
  // log("%u P-1 powerSmooth(B1=%u): %u bits\n", E, B1, u32(bits.size()));

  // Buffers used in both stages.
  Buffer bufTmp(makeBuf(context, BUF_RW, bufSize));
  Buffer bufAux(makeBuf(context, BUF_RW, bufSize));

  u32 maxBuffers = args.maxBuffers;

  if (!maxBuffers) {
    u32 nStage2Buffers = 5;
    maxBuffers = getAllocableBlocks(device, bufSize) - nStage2Buffers;
    log("%u P-1 GPU RAM fits %u stage2 buffers @ %.1f MB each\n", E, maxBuffers, bufSize/(1024.0f * 1024));
  }

  u32 nBufs = getNBuffers(maxBuffers);
  
  if (nBufs == 0) {
    log("%u P-1 stage2 not enough GPU RAM\n", E);
    throw "P-1 not enough memory";
  }    

  // Build the stage2 plan early (before stage1) in order to display plan stats at start.
  u32 nRounds = 2880 / nBufs;
  log("%u P-1 using %u stage2 buffers (%u rounds)\n", E, nBufs, nRounds);
  
  auto [startBlock, nPrimes, allSelected] = makePm1Plan(B1, B2);
  u32 nBlocks = allSelected.size();
  log("%u P-1 stage2: %u blocks starting at block %u (%u selected)\n",
          E, nBlocks, startBlock, nPrimes);

  // Start stage1 proper.
  log("%u P-1 starting stage1\n", E);
  {
    vector<u32> data((E - 1) / 32 + 1, 0);
    data[0] = 1;  
    writeData(data);
  }

  const u32 kEnd = bits.size();
  bool leadIn = true;
  TimeInfo timeInfo;

  Timer timer;

  assert(kEnd > 0);
  assert(!bits[kEnd - 1]);
  
  for (u32 k = 0; k < kEnd - 1; ++k) {
    bool doLog = (k == kEnd - 1) || ((k + 1) % 10000 == 0);
    
    bool leadOut = useLongCarry || doLog;
    coreStep(leadIn, leadOut, bits[k], bufAux, bufTmp, bufData);
    leadIn = leadOut;

    if ((k + 1) % 100 == 0 || doLog) {
      queue.finish();
      timeInfo.add(timer.deltaMillis(), (k + 1) - (k / 100) * 100);
      if (doLog) { doSmallLog(E, k + 1, dataResidue(), timeInfo, kEnd); }
    }
  }

  // See coreStep().
  if (leadIn) { fftP(bufData, bufAux); }
  tW(bufAux, bufTmp);
  tailFused(bufTmp);
  tH(bufTmp, bufAux);

  Buffer bufAcc(makeBuf(context, BUF_RW, bufSize));
  queue.copy<double>(bufAux, bufAcc, N);
  
  fftW(bufAux);
  carryA(bufAux, bufData);
  carryB(bufData);

  future<string> gcdFuture = async(launch::async, GCD, E, readData(), 1);
  bufAux.reset();
  
  Buffer bufA(makeBuf(context, BUF_RW, bufSize));
  Buffer bufBase(makeBuf(context, BUF_RW, bufSize));
  fftP(bufData, bufA);
  tW(bufA, bufBase);
  fftH(bufBase);
  
  Buffer bufB(makeBuf(context, BUF_RW, bufSize));
  Buffer bufC(makeBuf(context, BUF_RW, bufSize));

  if (getFreeMem(device) < u64(nBufs) * bufSize) {
    log("P-1 stage2 too little memory %u MB for %u buffers of %u b\n", u32(getFreeMem(device) / (1024 * 1024)), nBufs, bufSize);
    throw "P-1 not enough memory";
  }
  
  vector<Buffer> blockBufs;
  for (u32 i = 0; i < nBufs; ++i) { blockBufs.emplace_back(makeBuf(context, BUF_RW, bufSize)); }

  auto jset = getJset();

  Buffer bufBaseD2(makeBuf(context, BUF_RW, bufSize));
  exponentiate(bufBase, 30030*30030, bufTmp, bufBaseD2);
  
  for (u32 round = 0; round < nRounds; ++round) {
    timer.deltaSecs();

    exponentiate(bufBase, 8, bufTmp, bufA);    
    {
      u32 j0 = jset[round * nBufs + 0];
      assert(j0 & 1);
      exponentiate(bufBase, (j0 + 1) * 4, bufTmp, bufB);
      exponentiate(bufBase, j0 * j0, bufTmp, bufC);
      queue.copy<double>(bufC, blockBufs[0], N);
    }

    for (u32 i = 1; i < nBufs; ++i) {
      for (int steps = (jset[round * nBufs + i] - jset[round * nBufs + i - 1])/2; steps > 0; --steps) {
        multiplyLow(bufB, bufTmp, bufC);
        multiplyLow(bufA, bufTmp, bufB);
      }
      queue.copy<double>(bufC, blockBufs[i], N);
    }

    exponentiate(bufBaseD2, 2, bufTmp, bufA);                            // A := base^(2 * D^2)
    exponentiate(bufBaseD2, 2 * startBlock + 1, bufTmp, bufB);           // B := base^((2k + 1) * D^2)
    exponentiate(bufBaseD2, u64(startBlock) * startBlock, bufTmp, bufC); // C := base^(k^2 * D^2)

    queue.finish();
    float initSecs = timer.deltaSecs();
    // log("Round %u/%u: inited (%u buffers) in %u ms\n", round, nRounds, nBufs, timer.deltaMillis());

    u32 nSelected = 0;

    for (const auto& selected : allSelected) {
      for (u32 i = 0; i < nBufs; ++i) {
        if (selected[i + round * nBufs]) {
          ++nSelected;
          carryFused(bufAcc);
          tW(bufAcc, bufTmp);
          tailFusedMulDelta(bufTmp, bufC, blockBufs[i]);
          tH(bufTmp, bufAcc);
        }
      }
      multiplyLow(bufB, bufTmp, bufC);
      multiplyLow(bufA, bufTmp, bufB);
      queue.finish();
    }

    // queue.finish();
    log("Round %u of %u: init %.2f s; %.2f ms/mul; %u muls\n",
        round, nRounds, initSecs, (timer.deltaSecs() / nSelected) * 1000, nSelected);

    if (gcdFuture.valid()) {
      if (gcdFuture.wait_for(chrono::steady_clock::duration::zero()) == future_status::ready) {
        string gcd = gcdFuture.get();
        log("%u P-1 stage1 GCD: %s\n", E, gcd.empty() ? "no factor" : gcd.c_str());
        if (!gcd.empty()) { return gcd; }
      }
    }      
  }

  fftW(bufAcc);
  carryA(bufAcc, bufData);
  carryB(bufData);
  vector<u32> data = readData();

  if (gcdFuture.valid()) {
    gcdFuture.wait();
    string gcd = gcdFuture.get();
    log("%u P-1 stage1 GCD: %s\n", E, gcd.empty() ? "no factor" : gcd.c_str());
    if (!gcd.empty()) { return gcd; }
  }
  
  return data;
}

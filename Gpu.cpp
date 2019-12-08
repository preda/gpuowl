// Copyright (C) Mihai Preda.

#include "Gpu.h"
#include "ProofSet.h"
#include "Pm1Plan.h"
#include "checkpoint.h"
#include "state.h"
#include "Args.h"
#include "Signal.h"
#include "FFTConfig.h"
#include "GmpUtil.h"
#include "AllocTrac.h"
#include "Queue.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <future>
#include <optional>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#define TAU (2 * M_PIl)

static_assert(sizeof(double2) == 16, "size double2");
static_assert(sizeof(long double) > sizeof(double), "long double offers extended precision");

// Returns the primitive root of unity of order N, to the power k.
static double2 root1(u32 N, u32 k) {
  long double angle = - TAU / N * k;
  return double2{double(cosl(angle)), double(sinl(angle))};
}

static double2 *smallTrigBlock(u32 W, u32 H, double2 *p) {
  for (u32 line = 1; line < H; ++line) {
    for (u32 col = 0; col < W; ++col) {
      *p++ = root1(W * H, line * col);
    }
  }
  return p;
}

static ConstBuffer<double2> genSmallTrig(const Context& context, u32 size, u32 radix) {
  vector<double2> tab(size);
  auto *p = tab.data() + radix;
  u32 w = 0;
  for (w = radix; w < size; w *= radix) { p = smallTrigBlock(w, std::min(radix, size / w), p); }
  assert(p - tab.data() == size);
  return {context, "smallTrig", tab};
}

static u32 kAt(u32 H, u32 line, u32 col, u32 rep) {
  return (line + col * H) * 2 + rep;
}

static ConstBuffer<u32> genExtras(const Context& context, u32 E, u32 W, u32 H, u32 nW) {
  u32 N = 2u * W * H;
  vector<u32> extras;
  u32 groupWidth = W / nW;
  for (u32 line = 0; line < H; ++line) {
    for (u32 thread = 0; thread < groupWidth; ++thread) {
      extras.push_back(extra(N, E, kAt(H, line, thread, 0)));
    }
  }
  return {context, "extras", extras};
}

struct Weights {
  vector<double> aTab;
  vector<double> iTab;
  vector<double> groupWeights;
  vector<double> threadWeights;
  vector<u32> bits;
};

static long double weight(u32 N, u32 E, u32 H, u32 line, u32 col, u32 rep) {
  long double iN = 1 / (long double) N;
  // u32 k = (line + col * H) * 2 + rep;
  return exp2l(extra(N, E, kAt(H, line, col, rep)) * iN);
}

static long double invWeight(u32 N, u32 E, u32 H, u32 line, u32 col, u32 rep) {
  long double iN = 1 / (long double) N;
  return exp2l(- (extra(N, E, kAt(H, line, col, rep)) * iN));
}

static Weights genWeights(u32 E, u32 W, u32 H, u32 nW) {
  u32 N = 2u * W * H;

  vector<double> aTab, iTab;
  aTab.reserve(N);
  iTab.reserve(N);

  for (u32 line = 0; line < H; ++line) {
    for (u32 col = 0; col < W; ++col) {
      for (u32 rep = 0; rep < 2; ++rep) {
        auto a = weight(N, E, H, line, col, rep);
        auto ia = 1 / (4 * N * a);
        aTab.push_back(a);
        iTab.push_back(ia);
      }
    }
  }
  assert(aTab.size() == size_t(N) && iTab.size() == size_t(N));

  u32 groupWidth = W / nW;

  vector<double> groupWeights;
  for (u32 group = 0; group < H; ++group) {
    groupWeights.push_back(invWeight(N, E, H, group, 0, 0) / (4 * N));
    groupWeights.push_back(weight(N, E, H, group, 0, 0));
  }
  
  vector<double> threadWeights;
  for (u32 thread = 0; thread < groupWidth; ++thread) {
    threadWeights.push_back(invWeight(N, E, H, 0, thread, 0));
    threadWeights.push_back(weight(N, E, H, 0, thread, 0));
  }

  vector<u32> bits;
  double WEIGHT_STEP = weight(N, E, H, 0, 0, 1);
  double WEIGHT_BIGSTEP = weight(N, E, H, 0, groupWidth, 0);
  
  for (u32 line = 0; line < H; ++line) {
    for (u32 thread = 0; thread < groupWidth; ++thread) {
      std::bitset<32> b;
      double w = groupWeights[2*line+1] * threadWeights[2*thread+1];
      for (u32 block = 0; block < nW; ++block, w *= WEIGHT_BIGSTEP) {
        double w2 = w;
        if (w >= 2) { w *= 0.5; }
        for (u32 rep = 0; rep < 2; ++rep, w2 *= WEIGHT_STEP) {
          if (w2 >= 2) { b.set(block * 2 + rep); w2 *= 0.5; }
          if (isBigWord(N, E, kAt(H, line, block * groupWidth + thread, rep))) { b.set((nW + block) * 2 + rep); }
        }        
      }
      bits.push_back(b.to_ulong());
    }
  }

  return Weights{aTab, iTab, groupWeights, threadWeights, bits};
}

extern const char *CL_SOURCE;

static cl_program compile(const Args& args, cl_context context, u32 N, u32 E, u32 WIDTH, u32 SMALL_HEIGHT, u32 MIDDLE, u32 nW) {
  string clArgs = args.dump.empty() ? ""s : (" -save-temps="s + args.dump + "/" + numberK(N));
  vector<pair<string, std::any>> defines =
    {{"EXP", E},
     {"WIDTH", WIDTH},
     {"SMALL_HEIGHT", SMALL_HEIGHT},
     {"MIDDLE", MIDDLE},
     {"WEIGHT_STEP", double(weight(N, E, SMALL_HEIGHT * MIDDLE, 0, 0, 1))},
     {"IWEIGHT_STEP", double(invWeight(N, E, SMALL_HEIGHT * MIDDLE, 0, 0, 1))},
     {"WEIGHT_BIGSTEP", double(weight(N, E, SMALL_HEIGHT * MIDDLE, 0, WIDTH / nW, 0))},
     {"IWEIGHT_BIGSTEP", double(invWeight(N, E, SMALL_HEIGHT * MIDDLE, 0, WIDTH / nW, 0))},
    };
  
  for (const string& flag : args.flags) { defines.push_back(pair{flag, 1}); }
  
  cl_program program = compile({getDevice(args.device)}, context, CL_SOURCE, clArgs, defines);
  if (!program) { throw "OpenCL compilation"; }
  return program;
}

Gpu::Gpu(const Args& args, u32 E, u32 W, u32 BIG_H, u32 SMALL_H, u32 nW, u32 nH,
         cl_device_id device, bool timeKernels, bool useLongCarry, bool useMergedMiddle)
  : Gpu{args, E, W, BIG_H, SMALL_H, nW, nH, device, timeKernels, useLongCarry, useMergedMiddle, genWeights(E, W, BIG_H, nW)}
{}

Gpu::Gpu(const Args& args, u32 E, u32 W, u32 BIG_H, u32 SMALL_H, u32 nW, u32 nH,
         cl_device_id device, bool timeKernels, bool useLongCarry, bool useMergedMiddle, Weights&& weights) :
  E(E),
  N(W * BIG_H * 2),
  hN(N / 2),
  nW(nW),
  nH(nH),
  bufSize(N * sizeof(double)),
  useLongCarry(useLongCarry),
  useMiddle(BIG_H != SMALL_H),
  useMergedMiddle(BIG_H != SMALL_H && useMergedMiddle),
  timeKernels(timeKernels),
  device(device),
  context{device},
  program(compile(args, context.get(), N, E, W, SMALL_H, BIG_H / SMALL_H, nW)),
  queue(Queue::make(context, timeKernels, args.cudaYield)),

#define LOAD(name, workGroups) name(program.get(), queue, device, workGroups, #name)
  LOAD(carryFused, BIG_H + 1),
  LOAD(carryFusedMul, BIG_H + 1),
  LOAD(fftP, BIG_H),
  LOAD(fftW, BIG_H),
  LOAD(fftHin, (hN / SMALL_H)),
  LOAD(fftHout, (hN / SMALL_H)),
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
  LOAD(sum64, 256),
#undef LOAD

  bufTrigW{genSmallTrig(context, W, nW)},
  bufTrigH{genSmallTrig(context, SMALL_H, nH)},
  bufWeightA{context, "weightA", weights.aTab},
  bufWeightI{context, "weightI", weights.iTab},
  bufBits{context, "bits", weights.bits},
  bufExtras{genExtras(context, E, W, BIG_H, nW)},
  bufGroupWeights{context, "groupWeights", weights.groupWeights},
  bufThreadWeights{context, "threadWeights", weights.threadWeights},
  bufData{queue, "data", N},
  bufAux{queue, "aux", N},
  bufCheck{queue, "check", N},
  bufCarry{queue, "carry", N / 2},
  bufReady{queue, "ready", BIG_H},
  bufSmallOut{queue, "smallOut", 256},
  bufSumOut{queue, "sumOut", 1}
{
  // dumpBinary(program.get(), "isa.bin");
  program.reset();
  carryFused.setFixedArgs(2, bufCarry, bufReady, bufTrigW, bufBits, bufGroupWeights, bufThreadWeights);
  carryFusedMul.setFixedArgs(2, bufCarry, bufReady, bufTrigW, bufBits, bufGroupWeights, bufThreadWeights);
  fftP.setFixedArgs(2, bufWeightA, bufTrigW);
  fftW.setFixedArgs(2, bufTrigW);
  fftHin.setFixedArgs(2, bufTrigH);
  fftHout.setFixedArgs(1, bufTrigH);
    
  carryA.setFixedArgs(2, bufCarry, bufWeightI, bufExtras);
  carryM.setFixedArgs(2, bufCarry, bufWeightI, bufExtras);
  carryB.setFixedArgs(1, bufCarry, bufExtras);
  tailFused.setFixedArgs(2, bufTrigH);
  tailFusedMulDelta.setFixedArgs(4, bufTrigH);
  sum64.setFixedArgs(2, bufSumOut);
  
  queue->zero(bufReady, BIG_H);
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

vector<int> Gpu::readSmall(Buffer<int>& buf, u32 start) {
  readResidue(buf, bufSmallOut, start);
  return bufSmallOut.read(128);
}

unique_ptr<Gpu> Gpu::make(u32 E, const Args &args) {
  vector<FFTConfig> configs = FFTConfig::genConfigs();
        
  FFTConfig config = getFFTConfig(configs, E, args.fftSize);
  u32 WIDTH        = config.width;
  u32 SMALL_HEIGHT = config.height;
  u32 MIDDLE       = config.middle;
  u32 N = WIDTH * SMALL_HEIGHT * MIDDLE * 2;

  u32 nW = (WIDTH == 1024 || WIDTH == 256) ? 4 : 8;
  u32 nH = (SMALL_HEIGHT == 1024 || SMALL_HEIGHT == 256) ? 4 : 8;

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

  bool useMergedMiddle = 0;
  for (const string& flag : args.flags) if (flag == "MERGED_MIDDLE") useMergedMiddle = 1;

  if (useLongCarry) { log("using long carry kernels\n"); }

  bool timeKernels = args.timeKernels;

  return make_unique<Gpu>(args, E, WIDTH, SMALL_HEIGHT * MIDDLE, SMALL_HEIGHT, nW, nH,
                          getDevice(args.device), timeKernels, useLongCarry, useMergedMiddle);
}

vector<u32> Gpu::readAndCompress(ConstBuffer<int>& buf)  {
  // queue->zero(bufSumOut);
  while (true) {
    sum64(u32(buf.size * sizeof(int)), buf);
    vector<u64> expectedVect(1);
    bufSumOut >> expectedVect;
    vector<int> data = readOut(buf);
    u64 expectedSum = expectedVect[0];
    u64 sum = 0;
    for (auto it = data.begin(), end = data.end(); it < end; it += 2) {
      sum += u32(*it) | (u64(*(it + 1)) << 32);
    }
    if (sum != expectedSum) {
      log("GPU->host read failed (check %x vs %x)\n", unsigned(sum), unsigned(expectedSum));
    } else {
      return compactBits(std::move(data),  E);
    }
  }
}

vector<u32> Gpu::writeData(const vector<u32> &v) {
  writeIn(v, bufData);
  return v;
}

vector<u32> Gpu::writeCheck(const vector<u32> &v) {
  writeIn(v, bufCheck);
  return v;
}

// The modular multiplication io *= in.
void Gpu::modMul(Buffer<int>& in, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3, Buffer<int>& io, bool mul3) {
  fftP(in, buf1);
  tW(buf1, buf3);
  fftHin(buf3, buf1);		// GW:  buf1 could be reused, multiplier does not change -- pass in a leadIn argument???

  fftP(io, buf2);
  tW(buf2, buf3);
  fftHin(buf3, buf2);

  multiply(buf2, buf1);
  fftHout(buf2);

  tH(buf2, buf3);    
  fftW(buf3, buf2);
  mul3 ? carryM(buf2, io) : carryA(buf2, io);
  carryB(io);
};

void Gpu::writeState(const vector<u32> &check, u32 blockSize, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
  assert(blockSize > 0);
  writeCheck(check);
  bufData << bufCheck;
  bufAux  << bufCheck;

  u32 n = 0;
  for (n = 1; blockSize % (2 * n) == 0; n *= 2) {
    modSqLoop(n, buf1, buf2, bufData);
    modMul(bufAux, buf1, buf2, buf3, bufData);
    bufAux << bufData;
  }

  assert((n & (n - 1)) == 0);
  assert(blockSize % n == 0);
    
  blockSize /= n;
  assert(blockSize >= 2);
  
  for (u32 i = 0; i < blockSize - 2; ++i) {
    modSqLoop(n, buf1, buf2, bufData);
    modMul(bufAux, buf1, buf2, buf3, bufData);
  }
  
  modSqLoop(n, buf1, buf2, bufData);
  modMul(bufAux, buf1, buf2, buf3, bufData, true);
}

void Gpu::updateCheck(Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
  modMul(bufData, buf1, buf2, buf3, bufCheck);
}
  
bool Gpu::doCheck(u32 blockSize, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
  bufAux << bufCheck;
  modSqLoop(blockSize, buf1, buf2, bufAux, true);
  updateCheck(buf1, buf2, buf3);
  return equalNotZero(bufCheck, bufAux);
}

void Gpu::logTimeKernels() {
  if (timeKernels) {
    Queue::Profile profile = queue->getProfile();
    queue->clearProfile();
    double total = 0;
    for (auto& p : profile) { total += p.first.total; }
  
    for (auto& [stats, name]: profile) {
      float percent = 100 / total * stats.total;
      if (percent >= .01f) {
        log("%5.2f%% %-14s : %6.0f us/call x %5d calls\n",
            percent, name.c_str(), stats.total * (1e6f / stats.n), stats.n);
      }
    }
    log("Total time %.3f s\n", total);
  }
}

void Gpu::tW(Buffer<double>& in, Buffer<double>& out) {
  if (useMergedMiddle) fftMiddleIn(in, out);
  else {
    transposeW(in, out);
    if (useMiddle) fftMiddleIn(out, out);
  }
}

void Gpu::tH(Buffer<double>& in, Buffer<double>& out) {
  if (useMergedMiddle) fftMiddleOut(in, out);
  else {
    if (useMiddle) { fftMiddleOut(in, in); }
    transposeH(in, out);
  }
}
  
vector<int> Gpu::readOut(ConstBuffer<int> &buf) {
  transposeOut(buf, bufAux);
  return bufAux.read();
}

void Gpu::writeIn(const vector<u32>& words, Buffer<int>& buf) { writeIn(expandBits(words, N, E), buf); }

void Gpu::writeIn(const vector<int>& words, Buffer<int>& buf) {
  bufAux = words;
  transposeIn(bufAux, buf);
}

// io *= in; with buffers in low position.
void Gpu::multiplyLow(Buffer<double>& in, Buffer<double>& tmp, Buffer<double>& io) {
  multiply(io, in);
  fftHout(io);
  tH(io, tmp);
  carryFused(tmp, io);
  tW(io, tmp);
  fftHin(tmp, io);
}

// Auxiliary performing the top half of the cycle (excluding the bottom tailFused).
void Gpu::topHalf(Buffer<double>& in, Buffer<double>& out) {
  tH(in, out);
  carryFused(out, in);
  tW(in, out);
}

// See "left-to-right binary exponentiation" on wikipedia
// Computes out := base**exp
// All buffers are in "low" position.
void Gpu::exponentiate(const Buffer<double>& base, u64 exp, Buffer<double>& tmp, Buffer<double>& out) {
  if (exp == 0) {
    queue->zero(out, N / 2);
    u32 data = 1;
    fillBuf(queue->get(), out.get(), &data, sizeof(data));
    fftP(out, tmp);
    tW(tmp, out);
  } else {
    out << base;
    if (exp == 1) { return; }

    int p = 63;
    while ((exp >> p) == 0) { --p; }
    assert(p > 0);

    // square from "low" position.
    square(out);				// GW:  The multiply and square routines could also do fftHout
    fftHout(out);
    topHalf(out, tmp);

    while (true) {
      --p;
      if ((exp >> p) & 1) {
        fftHin(tmp, out); // to low

        // multiply from low
        multiply(out, base);			// GW:  The multiply and square routines could also do fftHout
	fftHout(out);
        topHalf(out, tmp);
      }
      if (p <= 0) { break; }

      // square
      tailFused(tmp, out);
      topHalf(out, tmp);
    }
  }

  fftHin(tmp, out); // to low
}

void Gpu::coreStep(bool leadIn, bool leadOut, bool mul3, Buffer<double>& buf1, Buffer<double>& bufTmp, Buffer<int>& io) {
  if (leadIn) { fftP(io, buf1); }
  tW(buf1, bufTmp);
  tailFused(bufTmp, buf1);
  tH(buf1, bufTmp);

  if (leadOut) {
    fftW(bufTmp, buf1);
    mul3 ? carryM(buf1, io) : carryA(buf1, io);
    carryB(io);
  } else {
    mul3 ? carryFusedMul(bufTmp, buf1) : carryFused(bufTmp, buf1);
  }  
}

void Gpu::modSqLoop(u32 reps, Buffer<double>& buf1, Buffer<double>& bufTmp, Buffer<int>& io, bool mul3) {
  bool leadIn = true;
  for (u32 i = 0; i < reps; ++i) {
    bool leadOut = useLongCarry || (i == reps - 1);
    coreStep(leadIn, leadOut, mul3 && (i == reps - 1), buf1, bufTmp, io);
    leadIn = leadOut;
  }
}

bool Gpu::equalNotZero(Buffer<int>& buf1, Buffer<int>& buf2) {
  queue->zero(bufSmallOut, 1);
  u32 sizeBytes = N * sizeof(int);
  isNotZero(sizeBytes, buf1, bufSmallOut);
  isEqual(sizeBytes, buf1, buf2, bufSmallOut);
  return bufSmallOut.read(1)[0];
}
  
u64 Gpu::bufResidue(Buffer<int> &buf) {
  u32 earlyStart = N/2 - 32;
  vector<int> readBuf = readSmall(buf, earlyStart);
  return residueFromRaw(N, E, readBuf);
}

static string getETA(u32 step, u32 total, float secsPerStep) {
  // assert(step <= total);
  int etaMins = (total - step) * secsPerStep * (1 / 60.f) + .5f;
  int days  = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins  = etaMins % 60;
  char buf[64];
  snprintf(buf, sizeof(buf), "%dd %02d:%02d", days, hours, mins);
  return string(buf);
}

static string makeLogStr(u32 E, string_view status, u32 k, u64 res, float secsPerIt, u32 nIters, double minBlockTime, double allMinBlockTime) {
  // float msPerSq = info.total / info.n;
  char buf[256];
  
  snprintf(buf, sizeof(buf), "%u %2s %8d %6.2f%%; %4.0f us/it (min %4.0f %4.0f); ETA %s; %s",
           E, status.data(), k, k / float(nIters) * 100,
           secsPerIt * 1'000'000, minBlockTime * 1'000'000, allMinBlockTime * 1'000'000, getETA(k, nIters, secsPerIt).c_str(),
           hex(res).c_str());
  return buf;
}

static void doBigLog(u32 E, u32 k, u64 res, bool checkOK, double secsPerIt, u32 nIters, u32 nErrors, double minBlockTime, double allMinBlockTime, double checkTime) {
  log("%s (check %.2fs)%s\n", makeLogStr(E, checkOK ? "OK" : "EE", k, res, secsPerIt, nIters, minBlockTime, allMinBlockTime).c_str(),
      checkTime, (nErrors ? " "s + to_string(nErrors) + " errors"s : ""s).c_str());
}

/*
static void doSmallLog(u32 E, u32 k, u64 res, double secsPerIt, u32 nIters, double minBlockTime, double allMinBlockTime) {
  log("%s\n", makeLogStr(E, "", k, res, secsPerIt, nIters, minBlockTime, allMinBlockTime).c_str());
}
*/

static void logPm1Stage1(u32 E, u32 k, u64 res, float secsPerIt, u32 nIters) {
  log("%s\n", makeLogStr(E, "P1", k, res, secsPerIt, nIters, 0, 0).c_str());
}

[[maybe_unused]] static void logPm1Stage2(u32 E, float ratioComplete) {
  char buf[256];
  snprintf(buf, sizeof(buf), "%u %2s %5.2f%%\n", E, "P2", ratioComplete * 100);
}

static bool equals9(const vector<u32> &a) {
  if (a[0] != 9) { return false; }
  for (auto it = next(a.begin()); it != a.end(); ++it) { if (*it) { return false; }}
  return true;
}

PRPState Gpu::loadPRP(u32 E, u32 iniBlockSize, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
  PRPState loaded(E, iniBlockSize);
  writeState(loaded.check, loaded.blockSize, buf1, buf2, buf3);

  u64 res64 = dataResidue();
  bool ok = (res64 == loaded.res64);
  updateCheck(buf1, buf2, buf3);

  std::string expected = " (expected "s + hex(loaded.res64) + ")";
  
  log("%u %2s %8d loaded: blockSize %d, %s%s\n",
      E, ok ? "OK" : "EE", loaded.k, loaded.blockSize, hex(res64).c_str(), ok ? "" : expected.c_str());

  if (!ok) { throw "error on load"; }

  return loaded;
}

static u32 mod3(const std::vector<u32> &words) {
  u32 r = 0;
  // uses the fact that 2**32 % 3 == 1.
  for (u32 w : words) { r += w % 3; }
  return r % 3;
}

static void doDiv3(int E, std::vector<u32> &words) {
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

void doDiv9(int E, std::vector<u32> &words) {
  doDiv3(E, words);
  doDiv3(E, words);
}

class IterationTimer {
  Timer timer;
  u32 kStart;

  double secsPerIt(double secs, u32 k) const { return secs / std::max(k - kStart, 1u); }
  
public:
  IterationTimer(u32 kStart) : kStart(kStart) {}
  
  // double at(u32 k) const { return secsPerIt(timer.elapsed(), k); }
  
  double reset(u32 k) {
    double secs = timer.deltaSecs();
    double ret = secsPerIt(secs, k);
    kStart = k;
    return ret;
  }
};

static u32 checkStepForErrors(u32 baseCheckStep, u32 nErrors) {
  if (baseCheckStep != 200'000) { return baseCheckStep; }
  
  switch (nErrors) {
    case 0: return 200'000;
    case 1: return 100'000;
    case 2: return  50'000;
    default: return 20'000;
  }
}

void Gpu::buildProof(u32 E, const Args& args) {
  u32 power = args.proofPow;
  assert(power);
  
}

tuple<bool, u64, u32> Gpu::isPrimePRP(u32 E, const Args &args) {  
  Buffer<double> buf1{queue, "buf1", N};
  Buffer<double> buf2{queue, "buf2", N};
  Buffer<double> buf3{queue, "buf3", N};
  
  u32 k = 0, blockSize = 0, nErrors = 0;

  {
    PRPState loaded = loadPRP(E, args.blockSize, buf1, buf2, buf3);
    k = loaded.k;
    blockSize = loaded.blockSize;
    nErrors = loaded.nErrors;
  }
  assert(blockSize > 0 && 10000 % blockSize == 0);
  
  const u32 kEnd = E; // Type-1 per http://www.mersenneforum.org/showpost.php?p=468378&postcount=209
  assert(k < kEnd);

  u32 checkStep = checkStepForErrors(args.logStep, nErrors);
  
  u32 startK = k;
  ProofSet proofSet{E, blockSize, args.proofPow};
  
  Signal signal;

  // Number of sequential errors (with no success in between). If this ever gets high enough, stop.
  int nSeqErrors = 0;

  bool isPrime = false;
  IterationTimer itTimer{startK};

  u64 finalRes64 = 0;
  u32 nTotalIters = ((kEnd - 1) / blockSize + 1) * blockSize;
  Timer timer;
  double minBlockTime = 1e9;
  double allMinBlockTime = 1e9;

  // string spinner = "|/-\\";
  string spinner = "-\\|/";
  size_t spinPos = 0;
  
  // future<void> saveFuture;
  while (true) {
    assert(k % blockSize == 0);
    assert(k < kEnd);
    
    u32 nextK = k + blockSize;
    timer.reset();
        
    if (nextK >= kEnd) {
      assert(kEnd > k);
      modSqLoop(kEnd - k, buf1, buf2, bufData);
      auto words = roundtripData();
      isPrime = equals9(words);
      doDiv9(E, words);
      finalRes64 = residue(words);
      log("%s %8d / %d, %s\n", isPrime ? "PP" : "CC", kEnd, E, hex(finalRes64).c_str());
      k = kEnd;
    }
    
    assert(nextK >= k);
    modSqLoop(nextK - k, buf1, buf2, bufData);
    k = nextK;

    // if (saveFuture.valid()) { saveFuture.get(); }

    bool persistProof = proofSet.shouldPersist(k);
    
    bool doStop = signal.stopRequested() || (args.iters && k - startK == args.iters);
    if (doStop) {
      log("Stopping, please wait..\n");
      signal.release();
    }

    bool doCheck = doStop || persistProof || (k % checkStep == 0) || (k >= kEnd && k < kEnd + blockSize) || (k - startK == 2 * blockSize);
    if (!doCheck) { this->updateCheck(buf1, buf2, buf3); }

    printf("\r%c", spinner[spinPos++]);
    fflush(stdout);
    if (spinPos >= spinner.size()) { spinPos = 0; }
    queue->finish();
    
    minBlockTime = std::min(minBlockTime, timer.deltaSecs());

    if (doCheck) {
      double timeExcludingCheck = itTimer.reset(k);
      u64 res64 = dataResidue();
      PRPState prpState{E, k, blockSize, res64, this->roundtripCheck(), nErrors};
      bool ok = this->doCheck(blockSize, buf1, buf2, buf3);
      
      if (ok) {
        if (k < kEnd) {          
          prpState.save(persistProof);
          // saveFuture = async(launch::async, &PRPState::save, std::move(prpState));
          
          allMinBlockTime = std::min(allMinBlockTime, minBlockTime);
          doBigLog(E, k, res64, ok, timeExcludingCheck, nTotalIters, nErrors, minBlockTime / blockSize, allMinBlockTime / blockSize, itTimer.reset(k));
        }
        assert(!isPrime || k >= kEnd);
        if (k >= kEnd) { return {isPrime, finalRes64, nErrors}; }
        nSeqErrors = 0;      
      } else {
        allMinBlockTime = std::min(allMinBlockTime, minBlockTime);
        doBigLog(E, k, res64, ok, timeExcludingCheck, nTotalIters, nErrors, minBlockTime / blockSize, allMinBlockTime / blockSize, itTimer.reset(k));

        ++nErrors;
        if (++nSeqErrors > 2) {
          log("%d sequential errors, will stop.\n", nSeqErrors);
          throw "too many errors";
        }
        checkStep = checkStepForErrors(args.logStep, nErrors);        
        auto loaded = loadPRP(E, blockSize, buf1, buf2, buf3);
        k = loaded.k;
        assert(blockSize == loaded.blockSize);
      }
      itTimer.reset(k);
      logTimeKernels();
      if (doStop) {
        // if (saveFuture.valid()) { saveFuture.get(); }
        throw "stop requested";
      }
      minBlockTime = 1e9;
    }
  }
}

bool isRelPrime(u32 D, u32 j);

struct SquaringSet {  
  std::string name;
  u32 N;
  Buffer<double> A, B, C;
  Gpu& gpu;

  SquaringSet(Gpu& gpu, u32 N, string_view name)
    : name(name)
    , N(N)
    , A{gpu.queue, this->name + ":A", N}
    , B{gpu.queue, this->name + ":B", N}
    , C{gpu.queue, this->name + ":C", N}
    , gpu(gpu)
  {}
   
  SquaringSet(const SquaringSet& rhs, string_view name) : SquaringSet{rhs.gpu, rhs.N, name} { copyFrom(rhs); }
  
  SquaringSet(Gpu& gpu, u32 N, const Buffer<double>& bufBase, Buffer<double>& bufTmp, array<u64, 3> exponents, string_view name)
    : SquaringSet(gpu, N, name) {
    
    gpu.exponentiate(bufBase, exponents[0], bufTmp, C);
    gpu.exponentiate(bufBase, exponents[1], bufTmp, B);
    if (exponents[2] == exponents[1]) {
      A << B;
    } else {
      gpu.exponentiate(bufBase, exponents[2], bufTmp, A);
    }
  }

  SquaringSet& operator=(const SquaringSet& rhs) {
    assert(N == rhs.N);
    copyFrom(rhs);
    return *this;
  }

  void step(Buffer<double>& bufTmp) {
    gpu.multiplyLow(B, bufTmp, C);
    gpu.multiplyLow(A, bufTmp, B);
  }

private:
  void copyFrom(const SquaringSet& rhs) {
    A << rhs.A;
    B << rhs.B;
    C << rhs.C;
  }
};

std::variant<string, vector<u32>> Gpu::factorPM1(u32 E, const Args& args, u32 B1, u32 B2) {
  assert(B1 && B2 && B2 >= B1);
  bufCheck.reset();

  if (!args.maxAlloc && !hasFreeMemInfo(device)) {
    log("%u P1 must specify -maxAlloc <MBytes> to limit GPU memory to use\n", E);
    throw("missing -maxAlloc");
  }
  
  vector<bool> bits = powerSmoothMSB(E, B1);

  // Buffers used in both stages.
  Buffer<double> bufTmp{queue, "tmp", N};
  Buffer<double> bufAux{queue, "aux", N};

  // --- Stage 1 ---

  u32 kBegin = 0;
  {
    P1State loaded{E, B1};
    assert(loaded.nBits == bits.size() || loaded.k == 0);
    assert(loaded.data.size() == (E - 1) / 32 + 1);
    writeData(loaded.data);
    kBegin = loaded.k;
  }

  const u32 kEnd = bits.size();
  log("%u P1 B1=%u, B2=%u; %u bits; starting at %u\n", E, B1, B2, kEnd, kBegin);

  Signal signal;
  // TimeInfo timeInfo;
  // Timer timer;
  Timer saveTimer;
  IterationTimer itTimer{kBegin};

  assert(kEnd > 0);
  assert(bits.front() && !bits.back());

  bool leadIn = true;
  for (u32 k = kBegin; k < kEnd - 1; ++k) {
    bool isAtEnd = k == kEnd - 2;
    bool doLog = (k + 1) % 10000 == 0; // || isAtEnd;
    bool doStop = signal.stopRequested();
    if (doStop) { log("Stopping, please wait..\n"); }
    bool doSave = doStop || saveTimer.elapsedSecs() > 300 || isAtEnd;
    bool leadOut = useLongCarry || doLog || doSave;
    coreStep(leadIn, leadOut, bits[k], bufAux, bufTmp, bufData);
    leadIn = leadOut;

    if ((k + 1) % 100 == 0 || doLog || doSave) {
      queue->finish();
      // timeInfo.add(timer.delta(), (k + 1) - (k / 100) * 100);
      if (doLog) {
        logPm1Stage1(E, k + 1, dataResidue(), itTimer.reset(k + 1), kEnd);
        logTimeKernels();
      }
      if (doSave) {
        P1State{E, B1, k + 1, u32(bits.size()), readData()}.save();
        // log("%u P1 saved at %u\n", E, k + 1);
        saveTimer.reset();
        if (doStop) { throw "stop requested"; }
        log("saved\n");
      }
    }
  }

  // See coreStep().
  if (leadIn) { fftP(bufData, bufAux); }

  HostAccessBuffer<double> bufAcc{queue, "acc", N};

  tW(bufAux, bufTmp);
  tailFused(bufTmp, bufAux);
  tH(bufAux, bufAcc);			// Save bufAcc for later use as an accumulator
  fftW(bufAcc, bufAux);
  carryA(bufAux, bufData);
  carryB(bufData);

  u32 beginPos = 0;
  {
    P2State loaded{E, B1, B2};
    if (loaded.k > 0) {
      if (loaded.raw.size() != N) {
        log("%u P2 wants %u words but savefile has %u\n", E, N, u32(loaded.raw.size()));
        throw "P2 savefile FFT size mismatch";
      }
      beginPos = loaded.k;
      bufAcc = loaded.raw;
      // queue->write(bufAcc, loaded.raw);
      log("%u P2 B1=%u, B2=%u, starting at %u\n", E, B1, B2, beginPos);
    }
  }

  future<string> gcdFuture;
  if (beginPos == 0) {
    gcdFuture = async(launch::async, GCD, E, readData(), 1);
    // timeInfo.add(timer.delta(), kEnd - (kEnd / 100) * 100);
    logPm1Stage1(E, kEnd, dataResidue(), itTimer.reset(kEnd), kEnd);
  }

  signal.release();
  
  // --- Stage 2 ---

  // Take bufData to "low" state stored in bufBase
  Buffer<double> bufBase{queue, "base", N};
  fftP(bufData, bufBase);
  tW(bufBase, bufTmp);
  fftHin(bufTmp, bufBase);
  
  auto [startBlock, nPrimes, allSelected] = makePm1Plan(B1, B2);
  assert(startBlock > 0);  
  u32 nBlocks = allSelected.size();
  log("%u P2 using blocks [%u - %u] to cover %u primes\n", E, startBlock, startBlock + nBlocks - 1, nPrimes);
  
  exponentiate(bufBase, 30030*30030, bufTmp, bufAux); // Aux := base^(D^2)

  constexpr auto jset = getJset();
  static_assert(jset[0] == 1);
  static_assert(jset[2880 - 1] == 15013);

  u32 beginJ = jset[beginPos];
  SquaringSet little{*this, N, bufBase, bufTmp, {beginJ*beginJ, 4 * (beginJ + 1), 8}, "little"};
  SquaringSet bigStart{*this, N, bufAux, bufTmp, {u64(startBlock)*startBlock, 2 * startBlock + 1, 2}, "bigStart"};
  bufBase.reset();
  bufAux.reset();
  SquaringSet big{*this, N, "big"};
  
  vector<Buffer<double>> blockBufs;
  try {
    bool hasMemInfo = hasFreeMemInfo(device);
    while ((blockBufs.size() < 2880 / 2) && (!hasMemInfo || (getFreeMem(device) >= 256 * 1024 * 1024))) {
      blockBufs.emplace_back(queue, "pm1BlockBuf", N);
    }
  } catch (const gpu_bad_alloc& e) {
  }

  vector<u32> stage2Data;
  
  if (blockBufs.empty()) {
    log("%u P2 Not enough GPU memory. Please wait for GCD\n", E);
  } else {
  
  u32 nBufs = blockBufs.size();
  log("%u P2 using %u buffers of %.1f MB each\n", E, nBufs, N / (1024.0f * 1024) * sizeof(double));
  
  queue->finish();
  logTimeKernels();
  Timer timer;
  
  u32 prevJ = jset[beginPos];
  for (u32 pos = beginPos; pos < 2880; pos += nBufs) {
    u32 nUsedBufs = min(nBufs, 2880 - pos);
    for (u32 i = 0; i < nUsedBufs; ++i) {
      int delta = jset[pos + i] - prevJ;
      prevJ = jset[pos + i];
      assert((delta & 1) == 0);
      for (int steps = delta / 2; steps > 0; --steps) { little.step(bufTmp); }
      blockBufs[i] << little.C;
    }

    queue->finish();
    // logTimeKernels();
    float setup = timer.deltaSecs();

    u32 nSelected = 0;
    bool first = true;
    for (const auto& selected : allSelected) {
      if (first) {
        big = bigStart;
        first = false;
      } else {
        big.step(bufTmp);
      }
      for (u32 i = 0; i < nUsedBufs; ++i) {
        if (selected[pos + i]) {
          ++nSelected;
          carryFused(bufAcc, bufTmp);
          tW(bufTmp, bufAcc);
          tailFusedMulDelta(bufAcc, bufTmp, big.C, blockBufs[i]);
          tH(bufTmp, bufAcc);
        }
      }
      queue->finish();
      // logTimeKernels();
    }

    if (pos + nBufs < 2880) { P2State{E, B1, B2, pos + nBufs, bufAcc.read()}.save(); }
    
    log("%u P2 %4u/2880: %u primes; setup %5.2f s, %7.3f ms/prime\n", E, pos + nUsedBufs, nSelected, setup, timer.deltaSecs() * 1000.f / (nSelected + 1));
    logTimeKernels();

    if (gcdFuture.valid() && gcdFuture.wait_for(chrono::steady_clock::duration::zero()) == future_status::ready) {
      string gcd = gcdFuture.get();
      log("%u P1 GCD: %s\n", E, gcd.empty() ? "no factor" : gcd.c_str());
      if (!gcd.empty()) { return gcd; }
    }      
  }

  fftW(bufAcc, bufTmp);
  carryA(bufTmp, bufData);
  carryB(bufData);
  stage2Data = readData();
  }

  if (gcdFuture.valid()) {
    string gcd = gcdFuture.get();
    log("%u P1 GCD: %s\n", E, gcd.empty() ? "no factor" : gcd.c_str());
    if (!gcd.empty()) { return gcd; }
  }
  
  return stage2Data;
}

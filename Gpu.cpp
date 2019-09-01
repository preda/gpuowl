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

static Buffer<double2> genSmallTrig(const Context& context, u32 size, u32 radix) {
  vector<double2> tab(size);
  auto *p = tab.data() + radix;
  u32 w = 0;
  for (w = radix; w < size; w *= radix) { p = smallTrigBlock(w, std::min(radix, size / w), p); }
  assert(p - tab.data() == size);
  return context.constBuf(tab);
}

static u32 kAt(u32 H, u32 line, u32 col, u32 rep) {
  return (line + col * H) * 2 + rep;
}

static Buffer<u32> genExtras(const Context& context, u32 E, u32 W, u32 H, u32 nW) {
  u32 N = 2u * W * H;
  vector<u32> extras;
  u32 groupWidth = W / nW;
  for (u32 line = 0; line < H; ++line) {
    for (u32 thread = 0; thread < groupWidth; ++thread) {
      extras.push_back(extra(N, E, kAt(H, line, thread, 0)));
    }
  }
  return context.constBuf(extras);
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
  program(compile(args, context.get(), N, E, W, SMALL_H, BIG_H / SMALL_H, nW)),
  queue(makeQueue(device, context.get())),  

#define LOAD(name, workGroups) name(program.get(), queue, device, workGroups, #name, timeKernels)
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

  bufTrigW{genSmallTrig(context, W, nW)},
  bufTrigH{genSmallTrig(context, SMALL_H, nH)},
  bufExtras{genExtras(context, E, W, BIG_H, nW)},
  
  bufData{context.hostAccessBuf<decltype(bufData)::type>(N)},
  bufAux{context.hostAccessBuf<decltype(bufAux)::type>(N)},
  
  bufCarry{context.buffer<decltype(bufCarry)::type>(N / 2)},
  bufReady{context.buffer<decltype(bufReady)::type>(BIG_H)},
  bufSmallOut(context.hostAccessBuf<decltype(bufSmallOut)::type>(256))
{
  // dumpBinary(program.get(), "isa.bin");
  program.reset();

  Weights weights = genWeights(E, W, BIG_H, nW);  
  bufWeightA = context.constBuf(weights.aTab);
  bufWeightI = context.constBuf(weights.iTab);

  bufGroupWeights = context.constBuf(weights.groupWeights);
  bufThreadWeights = context.constBuf(weights.threadWeights);
  bufBits = context.constBuf(weights.bits);
  
  carryFused.setFixedArgs(   1, bufCarry, bufReady, bufTrigW, bufBits, bufGroupWeights, bufThreadWeights);
  carryFusedMul.setFixedArgs(1, bufCarry, bufReady, bufWeightA, bufWeightI, bufTrigW, bufExtras);
  fftP.setFixedArgs(2, bufWeightA, bufTrigW);
  fftW.setFixedArgs(1, bufTrigW);
  fftH.setFixedArgs(1, bufTrigH);
    
  carryA.setFixedArgs(2, bufCarry, bufWeightI, bufExtras);
  carryM.setFixedArgs(2, bufCarry, bufWeightI, bufExtras);
  carryB.setFixedArgs(1, bufCarry, bufExtras);
  tailFused.setFixedArgs(1, bufTrigH);
  tailFusedMulDelta.setFixedArgs(3, bufTrigH);
    
  queue.zero(bufReady, BIG_H);
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

vector<int> Gpu::readSmall(Buffer<int>& buf, u32 start) {
  readResidue(buf, bufSmallOut, start);
  return queue.read<int>(bufSmallOut, 128);                    
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
void Gpu::modMul(Buffer<int>& in, bool mul3, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3, Buffer<int>& io) {
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

void Gpu::writeState(const vector<u32> &check, u32 blockSize, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
  assert(blockSize > 0);
    
  writeCheck(check);
  queue.copyFromTo(bufCheck, bufData);
  queue.copyFromTo(bufCheck, bufAux);

  u32 n = 0;
  for (n = 1; blockSize % (2 * n) == 0; n *= 2) {
    modSqLoop(n, false, buf1, buf2, bufData);  // dataLoop(n);
    modMul(bufAux, false, buf1, buf2, buf3, bufData);
    queue.copyFromTo(bufData, bufAux);
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

void Gpu::updateCheck(Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
  modMul(bufData, false, buf1, buf2, buf3, bufCheck);
}
  
bool Gpu::doCheck(u32 blockSize, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
  queue.copyFromTo(bufCheck, bufAux);
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

void Gpu::tW(Buffer<double>& in, Buffer<double>& out) {
  transposeW(in, out);
  if (useMiddle) { fftMiddleIn(out); }
}

void Gpu::tH(Buffer<double>& in, Buffer<double>& out) {
  if (useMiddle) { fftMiddleOut(in); }
  transposeH(in, out);
}
  
vector<int> Gpu::readOut(Buffer<int> &buf) {
  transposeOut(buf, bufAux);
  return queue.read(bufAux);
}

void Gpu::writeIn(const vector<u32>& words, Buffer<int>& buf) { writeIn(expandBits(words, N, E), buf); }

void Gpu::writeIn(const vector<int>& words, Buffer<int>& buf) {
  queue.write(bufAux, words);
  transposeIn(bufAux, buf);
}

// io *= in; with buffers in low position.
void Gpu::multiplyLow(Buffer<double>& in, Buffer<double>& tmp, Buffer<double>& io) {
  multiply(io, in);
  fftH(io);
  tH(io, tmp);
  carryFused(tmp);
  tW(tmp, io);
  fftH(io);
}

// Auxiliary performing the top half of the cycle (excluding the buttom tailFused).
void Gpu::topHalf(Buffer<double>& tmp, Buffer<double>& io) {
  tH(io, tmp);
  carryFused(tmp);
  tW(tmp, io);
}

// See "left-to-right binary exponentiation" on wikipedia
// Computes out := base**exp
// All buffers are in "low" position.
void Gpu::exponentiate(Buffer<double>& base, u64 exp, Buffer<double>& tmp, Buffer<double>& out) {
  if (exp == 0) {
    queue.zero(out, N * sizeof(u32));
    u32 data = 1;
    fillBuf(queue.get(), out.get(), &data, sizeof(data));
    // write(queue.get(), false, out, sizeof(u32), &data);
    // queue.writeAsync(out, vector<>{1});    
    fftP(out, tmp);
    tW(tmp, out);    
  } else {
    queue.copyFromTo(base, out);
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

void Gpu::coreStep(bool leadIn, bool leadOut, bool mul3, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<int>& io) {
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

void Gpu::modSqLoop(u32 reps, bool mul3, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<int>& io) {
  assert(reps > 0);
  bool leadIn = true;
        
  for (u32 i = 0; i < reps; ++i) {
    bool leadOut = useLongCarry || (i == reps - 1);
    coreStep(leadIn, leadOut, mul3 && (i == reps - 1), buf1, buf2, io);
    leadIn = leadOut;
  }
}

bool Gpu::equalNotZero(Buffer<int>& buf1, Buffer<int>& buf2) {
  queue.zero(bufSmallOut, 1);
  u32 sizeBytes = N * sizeof(int);
  isNotZero(sizeBytes, buf1, bufSmallOut);
  isEqual(sizeBytes, buf1, buf2, bufSmallOut);
  return queue.read(bufSmallOut, 1)[0];
}
  
u64 Gpu::bufResidue(Buffer<int> &buf) {
  u32 earlyStart = N/2 - 32;
  vector<int> readBuf = readSmall(buf, earlyStart);
  return residueFromRaw(N, E, readBuf);
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

static string makeLogStr(u32 E, string status, u32 k, u64 res, TimeInfo info, u32 nIters) {
  float msPerSq = info.total / info.n;
  char buf[256];
  string ghzStr;
  
  snprintf(buf, sizeof(buf), "%u %2s %8d %5.2f%%; %4d us/sq;%s ETA %s; %016llx",
           E, status.c_str(), k, k / float(nIters) * 100,
           int(msPerSq*1000+0.5f), ghzStr.c_str(), getETA(k, nIters, msPerSq).c_str(), res);
  return buf;
}

static void doLog(u32 E, u32 k, u32 timeCheck, u64 res, bool checkOK, TimeInfo &stats, u32 nIters) {
  log("%s (check %.2fs)\n",      
      makeLogStr(E, checkOK ? "OK" : "EE", k, res, stats, nIters).c_str(),
      timeCheck * .001f);
  stats.reset();
}

static void doSmallLog(u32 E, u32 k, u64 res, TimeInfo &stats, u32 nIters) {
  log("%s\n", makeLogStr(E, "", k, res, stats, nIters).c_str());
  stats.reset();
}

static bool equals9(const vector<u32> &a) {
  if (a[0] != 9) { return false; }
  for (auto it = next(a.begin()); it != a.end(); ++it) { if (*it) { return false; }}
  return true;
}

PRPState Gpu::loadPRP(u32 E, u32 iniBlockSize, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
  auto loaded = PRPState::load(E, iniBlockSize);

  writeState(loaded.check, loaded.blockSize, buf1, buf2, buf3);

  u64 res64 = dataResidue();
  bool ok = (res64 == loaded.res64);
  updateCheck(buf1, buf2, buf3);
  if (!ok) {
#ifdef __MINGW64__
    log("%u EE loaded: %d, blockSize %d, %016I64x (expected %016I64x)\n", E, loaded.k, loaded.blockSize, res64, loaded.res64);
#else
    log("%u EE loaded: %d, blockSize %d, %016llx (expected %016llx)\n",  E, loaded.k, loaded.blockSize, res64, loaded.res64);
#endif          
    throw "error on load";
  }

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

pair<bool, u64> Gpu::isPrimePRP(u32 E, const Args &args) {
  auto buf1{context.buffer<double>(N)};
  auto buf2{context.buffer<double>(N)};
  auto buf3{context.buffer<double>(N)};
  
  bufCheck = context.hostAccessBuf<decltype(bufCheck)::type>(N);
  
  PRPState loaded = loadPRP(E, args.blockSize, buf1, buf2, buf3);

  u32 k = loaded.k;
  u32 blockSize = loaded.blockSize;
  assert(blockSize > 0 && 10000 % blockSize == 0);
  
  const u32 kEnd = E; // Type-1 per http://www.mersenneforum.org/showpost.php?p=468378&postcount=209
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
      isPrime = equals9(words);
      doDiv9(E, words);
      finalRes64 = residue(words);

#ifdef __MINGW64__
      log("%s %8d / %d, %016I64x\n", isPrime ? "PP" : "CC", kEnd, E, finalRes64);
#else
      log("%s %8d / %d, %016llx\n", isPrime ? "PP" : "CC", kEnd, E, finalRes64);
#endif

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

    if (args.iters && k - startK == args.iters) { doStop = true; }

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

std::variant<string, vector<u32>> Gpu::factorPM1(u32 E, const Args& args, u32 B1, u32 B2) {
  assert(B1 && B2 && B2 > B1);
  
  vector<bool> bits = powerSmoothBitsRev(E, B1);
  // log("%u P-1 powerSmooth(B1=%u): %u bits\n", E, B1, u32(bits.size()));

  // Buffers used in both stages.
  auto bufTmp{context.buffer<double>(N)};
  auto bufAux{context.buffer<double>(N)};

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

  auto bufAcc{context.buffer<double>(N)};
  queue.copyFromTo(bufAux, bufAcc);
  
  fftW(bufAux);
  carryA(bufAux, bufData);
  carryB(bufData);

  future<string> gcdFuture = async(launch::async, GCD, E, readData(), 1);
  bufAux.reset();
  
  auto bufA{context.buffer<double>(N)};
  auto bufBase{context.buffer<double>(N)};
  fftP(bufData, bufA);
  tW(bufA, bufBase);
  fftH(bufBase);
  
  auto bufB{context.buffer<double>(N)};
  auto bufC{context.buffer<double>(N)};

  if (getFreeMem(device) < u64(nBufs) * bufSize) {
    log("P-1 stage2 too little memory %u MB for %u buffers of %u b\n", u32(getFreeMem(device) / (1024 * 1024)), nBufs, bufSize);
    throw "P-1 not enough memory";
  }
  
  vector<Buffer<double>> blockBufs;
  for (u32 i = 0; i < nBufs; ++i) { blockBufs.push_back(context.buffer<double>(N)); }

  auto jset = getJset();

  auto bufBaseD2{context.buffer<double>(N)};
  exponentiate(bufBase, 30030*30030, bufTmp, bufBaseD2);
  
  for (u32 round = 0; round < nRounds; ++round) {
    timer.deltaSecs();

    exponentiate(bufBase, 8, bufTmp, bufA);    
    {
      u32 j0 = jset[round * nBufs + 0];
      assert(j0 & 1);
      exponentiate(bufBase, (j0 + 1) * 4, bufTmp, bufB);
      exponentiate(bufBase, j0 * j0, bufTmp, bufC);
      queue.copyFromTo(bufC, blockBufs[0]);
    }

    for (u32 i = 1; i < nBufs; ++i) {
      for (int steps = (jset[round * nBufs + i] - jset[round * nBufs + i - 1])/2; steps > 0; --steps) {
        multiplyLow(bufB, bufTmp, bufC);
        multiplyLow(bufA, bufTmp, bufB);
      }
      queue.copyFromTo(bufC, blockBufs[i]);
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

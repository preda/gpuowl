// Copyright Mihai Preda and George Woltman.

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

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <algorithm>
#include <future>
#include <optional>
#include <numeric>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static_assert(sizeof(double2) == 16, "size double2");
static_assert(sizeof(long double) > sizeof(double), "long double offers extended precision");

// Returns the primitive root of unity of order N, to the power k.
static double2 root1(u32 N, u32 k) {
  assert(k < N);
  if (k >= N/2) {
    auto [c, s] = root1(N, k - N/2);
    return {-c, -s};
  } else if (k > N/4) {
    auto [c, s] = root1(N, N/2 - k);
    return {-c, s};
  } else if (k > N/8) {
    auto [c, s] = root1(N, N/4 - k);
    return {-s, -c};
  } else {
    assert(!(N&7));
    assert(k <= N/8);
    N /= 2;
    long double angle = - M_PIl * k / N;
    return {cosl(angle), sinl(angle)};    
  }
}

static double2 *smallTrigBlock(u32 W, u32 H, double2 *p) {
  for (u32 line = 1; line < H; ++line) {
    for (u32 col = 0; col < W; ++col) {
      *p++ = root1(W * H, line * col);
      // if(abs((p-1)->first) < 1e-16) { printf("%u %u %u %u %g\n", line, col, W, H, (p-1)->first); }
    }
  }
  return p;
}

static ConstBuffer<double2> genSmallTrig(const Context& context, u32 size, u32 radix) {
  vector<double2> tab(size);
  auto *p = tab.data() + radix;
  for (u32 w = radix; w < size; w *= radix) { p = smallTrigBlock(w, std::min(radix, size / w), p); }
  assert(p - tab.data() == size);
  return {context, "smallTrig", tab};
}

static ConstBuffer<double2> genMiddleTrig(const Context& context, u32 smallH, u32 middle) {
  vector<double2> tab;
  if (middle == 1) {
    tab.resize(1);
  } else {  
    u32 size = smallH * (middle - 1);
    tab.resize(size);
    auto *p = smallTrigBlock(smallH, middle, tab.data());
    assert(p - tab.data() == size);
  }
  return {context, "middleTrig", tab};
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
  return exp2l(extra(N, E, kAt(H, line, col, rep)) * iN);
}

static long double invWeight(u32 N, u32 E, u32 H, u32 line, u32 col, u32 rep) {
  long double iN = 1 / (long double) N;
  return exp2l(- (extra(N, E, kAt(H, line, col, rep)) * iN));
}

static double boundUnderOne(double x) { return std::min(x, nexttoward(1, 0)); }

static Weights genWeights(u32 E, u32 W, u32 H, u32 nW) {
  u32 N = 2u * W * H;

  vector<double> aTab, iTab;
  aTab.reserve(N);
  iTab.reserve(N);

  for (u32 line = 0; line < H; ++line) {
    for (u32 col = 0; col < W; ++col) {
      for (u32 rep = 0; rep < 2; ++rep) {
        long double a = weight(N, E, H, line, col, rep);
        // Double the weight and inverse weight so that optionalHalve and optionalDouble can save one instruction
        aTab.push_back(2.0 * a);
        iTab.push_back(2.0 * boundUnderOne(1 / a));
      }
    }
  }
  assert(aTab.size() == size_t(N) && iTab.size() == size_t(N));

  u32 groupWidth = W / nW;

  vector<double> groupWeights;
  for (u32 group = 0; group < H; ++group) {
    long double w = weight(N, E, H, group, 0, 0);
    // Double the weight and inverse weight so that optionalHalve and optionalDouble can save one instruction
    groupWeights.push_back(2.0 * boundUnderOne(1.0 / w));
    groupWeights.push_back(2.0 * w);
  }
  
  vector<double> threadWeights;
  for (u32 thread = 0; thread < groupWidth; ++thread) {
    threadWeights.push_back(invWeight(N, E, H, 0, thread, 0) - 1.0);
    threadWeights.push_back(weight(N, E, H, 0, thread, 0) - 1.0);
  }

  vector<u32> bits;
  
  for (u32 line = 0; line < H; ++line) {
    for (u32 thread = 0; thread < groupWidth; ) {
      std::bitset<32> b;
      for (u32 bitoffset = 0; bitoffset < 32; bitoffset += nW*2, ++thread) {
        for (u32 block = 0; block < nW; ++block) {
          for (u32 rep = 0; rep < 2; ++rep) {
            if (isBigWord(N, E, kAt(H, line, block * groupWidth + thread, rep))) { b.set(bitoffset + block * 2 + rep); }
          }        
	}
      }
      bits.push_back(b.to_ulong());
    }
  }

  return Weights{aTab, iTab, groupWeights, threadWeights, bits};
}


extern const char *CL_SOURCE;

namespace {

string toLiteral(u32 value) { return to_string(value) + 'u'; }
string toLiteral(i32 value) { return to_string(value); }
[[maybe_unused]] string toLiteral(u64 value) { return to_string(value) + "ul"; }
string toLiteral(double value) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%a", value);
  return buf;
}

struct Define {
  const string str;

  template<typename T> Define(const string& label, T value) : str{label + '=' + toLiteral(value)} {
    assert(label.find('=') == string::npos);
  }

  Define(const string& labelAndVal) : str{labelAndVal} {
    assert(labelAndVal.find('=') != string::npos);
  }
  

  operator string() const { return str; }
};

cl_program compile(const Args& args, cl_context context, u32 N, u32 E, u32 WIDTH, u32 SMALL_HEIGHT, u32 MIDDLE, u32 nW, bool isPm1) {
  string clArgs = args.dump.empty() ? ""s : (" -save-temps="s + args.dump + "/" + numberK(N));
  if (!args.safeMath) { clArgs += " -cl-unsafe-math-optimizations"; }

  vector<Define> defines =
    {{"EXP", E},
     {"WIDTH", WIDTH},
     {"SMALL_HEIGHT", SMALL_HEIGHT},
     {"MIDDLE", MIDDLE},
     {"PM1", (isPm1 ? 1 : 0)},
    };

  cl_device_id id = getDevice(args.device);
  if (isAmdGpu(id)) { defines.push_back({"AMDGPU", 1}); }

  // if PRP force carry64 when carry32 might exceed 0x70000000
  // if P-1 force carry64 when carry32 might exceed a very conservative 0x6C000000
  // when using carryFusedMul during P-1 mul-by-3, force carry64 when carry32 might exceed 0x6C000000 / 3.
  if (FFTConfig::getMaxCarry32(N, E) > (isPm1 ? 0x6C00 : 0x7000)) { defines.push_back({"CARRY64", 1}); }
  if (isPm1 && FFTConfig::getMaxCarry32(N, E) > 0x6C00 / 3) { defines.push_back({"CARRYM64", 1}); }

  // If we are near the maximum exponent for this FFT, then we may need to set some chain #defines
  // to reduce the round off errors.
  auto [max_accuracy, mm_chain, mm2_chain, ultra_trig] = FFTConfig::getChainLengths(N, E, MIDDLE);
  if (mm_chain) { defines.push_back({"MM_CHAIN", mm_chain}); }
  if (mm2_chain) { defines.push_back({"MM2_CHAIN", mm2_chain}); }
  if (max_accuracy) { defines.push_back({"MAX_ACCURACY", 1}); }
  if (ultra_trig) { defines.push_back({"ULTRA_TRIG", 1}); }

  defines.push_back({"WEIGHT_STEP_MINUS_1", double(weight(N, E, SMALL_HEIGHT * MIDDLE, 0, 0, 1) - 1)});
  defines.push_back({"IWEIGHT_STEP_MINUS_1", double(invWeight(N, E, SMALL_HEIGHT * MIDDLE, 0, 0, 1) - 1)});

  string clSource = CL_SOURCE;
  for (const string& flag : args.flags) {
    auto pos = flag.find('=');
    string label = (pos == string::npos) ? flag : flag.substr(0, pos);
    if (clSource.find(label) == string::npos) {
      log("%s not used\n", label.c_str());
      throw "-use with unknown key";
    }
    if (pos == string::npos) {
      defines.push_back({label, 1});
    } else {
      defines.push_back(flag);
    }
  }

  vector<string> strDefines;
  strDefines.insert(strDefines.begin(), defines.begin(), defines.end());

  cl_program program{};
  if (args.binaryFile.empty()) {
    program = compile(context, id, CL_SOURCE, clArgs, strDefines);
  } else {
    program = loadBinary(context, id, args.binaryFile);
  }
  if (!program) { throw "OpenCL compilation"; }
  // dumpBinary(program, "dump.bin");
  return program;
}

}

Gpu::Gpu(const Args& args, u32 E, u32 W, u32 BIG_H, u32 SMALL_H, u32 nW, u32 nH,
         cl_device_id device, bool timeKernels, bool useLongCarry, bool isPm1)
  : Gpu{args, E, W, BIG_H, SMALL_H, nW, nH, device, timeKernels, useLongCarry, genWeights(E, W, BIG_H, nW), isPm1}
{}

Gpu::Gpu(const Args& args, u32 E, u32 W, u32 BIG_H, u32 SMALL_H, u32 nW, u32 nH,
         cl_device_id device, bool timeKernels, bool useLongCarry, Weights&& weights, bool isPm1) :
  E(E),
  N(W * BIG_H * 2),
  hN(N / 2),
  nW(nW),
  nH(nH),
  bufSize(N * sizeof(double)),
  WIDTH(W),
  useLongCarry(useLongCarry),
  timeKernels(timeKernels),
  device(device),
  context{device},
  program(compile(args, context.get(), N, E, W, SMALL_H, BIG_H / SMALL_H, nW, isPm1)),
  queue(Queue::make(context, timeKernels, args.cudaYield)),

  // Specifies size in number of workgroups
#define LOAD(name, nGroups) name{program.get(), queue, device, nGroups, #name}
  // Specifies size in "work size": workSize == nGroups * groupSize
#define LOAD_WS(name, workSize) name{program.get(), queue, device, #name, workSize}
  
  LOAD(carryFused,    BIG_H + 1),
  LOAD(carryFusedMul, BIG_H + 1),
  LOAD(fftP, BIG_H),
  LOAD(fftW,   BIG_H),
  LOAD(fftHin,  hN / SMALL_H),
  LOAD(fftHout, hN / SMALL_H),
  LOAD_WS(fftMiddleIn,  hN / (BIG_H / SMALL_H)),
  LOAD_WS(fftMiddleOut, hN / (BIG_H / SMALL_H)),
  LOAD_WS(carryA,  hN / 16),
  LOAD_WS(carryM,  hN / 16),
  LOAD_WS(carryB,  hN / 16),
  LOAD(transposeW,   (W/64) * (BIG_H/64)),
  LOAD(transposeH,   (W/64) * (BIG_H/64)),
  LOAD(transposeIn,  (W/64) * (BIG_H/64)),
  LOAD(transposeOut, (W/64) * (BIG_H/64)),
  LOAD(multiply,      hN / SMALL_H),
  LOAD(multiplyDelta, hN / SMALL_H),
  LOAD(tailFusedSquare,   hN / SMALL_H / 2),
  LOAD(tailFusedMulDelta, hN / SMALL_H / 2),
  LOAD(tailFusedMulLow,   hN / SMALL_H / 2),
  LOAD(tailFusedMul,      hN / SMALL_H / 2),
  LOAD(tailSquareLow,     hN / SMALL_H / 2),
  LOAD(tailMulLowLow,     hN / SMALL_H / 2),
  LOAD(readResidue, 1),
  LOAD(isNotZero, 256),
  LOAD(isEqual, 256),
  LOAD(sum64, 256),
#undef LOAD_WS
#undef LOAD

  bufTrigW{genSmallTrig(context, W, nW)},
  bufTrigH{genSmallTrig(context, SMALL_H, nH)},
  bufTrigM{genMiddleTrig(context, SMALL_H, BIG_H / SMALL_H)},
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
  bufRoundoff{queue, "roundoff", 8 + 1024 * 1024},
  bufCarryMax{queue, "carryMax", 8},
  bufCarryMulMax{queue, "carryMulMax", 8},
  bufSmallOut{queue, "smallOut", 256},
  bufSumOut{queue, "sumOut", 1},
  buf1{queue, "buf1", N},
  buf2{queue, "buf2", N},
  buf3{queue, "buf3", N},
  args{args}
{
  // dumpBinary(program.get(), "isa.bin");
  program.reset();
  carryFused.setFixedArgs(  2, bufCarry, bufReady, bufTrigW, bufBits, bufGroupWeights, bufThreadWeights, bufRoundoff, bufCarryMax);
  carryFusedMul.setFixedArgs(2, bufCarry, bufReady, bufTrigW, bufBits, bufGroupWeights, bufThreadWeights, bufRoundoff, bufCarryMulMax);
  fftP.setFixedArgs(2, bufWeightA, bufTrigW);
  fftW.setFixedArgs(2, bufTrigW);
  fftHin.setFixedArgs(2, bufTrigH);
  fftHout.setFixedArgs(1, bufTrigH);
  fftMiddleIn.setFixedArgs(2, bufTrigM);
  fftMiddleOut.setFixedArgs(2, bufTrigM);
    
  carryA.setFixedArgs( 2, bufCarry, bufWeightI, bufExtras, bufRoundoff, bufCarryMax);
  carryM.setFixedArgs(2, bufCarry, bufWeightI, bufExtras, bufRoundoff, bufCarryMulMax);
  carryB.setFixedArgs(1, bufCarry, bufExtras);

  tailFusedMulDelta.setFixedArgs(4, bufTrigH, bufTrigH);
  tailFusedMulLow.setFixedArgs(3, bufTrigH, bufTrigH);
  tailFusedMul.setFixedArgs(3, bufTrigH, bufTrigH);
  tailMulLowLow.setFixedArgs(2, bufTrigH);
  
  tailFusedSquare.setFixedArgs(2, bufTrigH, bufTrigH);
  tailSquareLow.setFixedArgs(2, bufTrigH, bufTrigH);

  bufReady.zero();
  bufRoundoff.zero();
  bufCarryMax.zero();
  bufCarryMulMax.zero();
}

vector<Buffer<i32>> Gpu::makeBufVector(u32 size) {
  vector<Buffer<i32>> r;
  for (u32 i = 0; i < size; ++i) { r.emplace_back(queue, "vector", N); }
  return r;
}

static FFTConfig getFFTConfig(u32 E, string fftSpec) {
  if (fftSpec.empty()) {
    vector<FFTConfig> configs = FFTConfig::genConfigs();
    for (FFTConfig c : configs) { if (c.maxExp() >= E) { return c; } }
    log("No FFT for exponent %u\n", E);
    throw "No FFT for exponent";
  }
  return FFTConfig::fromSpec(fftSpec);
}

vector<int> Gpu::readSmall(Buffer<int>& buf, u32 start) {
  readResidue(bufSmallOut, buf, start);
  return bufSmallOut.read(128);
}

unique_ptr<Gpu> Gpu::make(u32 E, const Args &args, bool isPm1) {        
  FFTConfig config = getFFTConfig(E, args.fftSpec);
  u32 WIDTH        = config.width;
  u32 SMALL_HEIGHT = config.height;
  u32 MIDDLE       = config.middle;
  u32 N = WIDTH * SMALL_HEIGHT * MIDDLE * 2;

  u32 nW = (WIDTH == 1024 || WIDTH == 256) ? 4 : 8;
  u32 nH = (SMALL_HEIGHT == 1024 || SMALL_HEIGHT == 256) ? 4 : 8;

  float bitsPerWord = E / float(N);
  log("%u FFT: %s %s (%.2f bpw)\n", E, numberK(N).c_str(), config.spec().c_str(), bitsPerWord);

  if (bitsPerWord > 20) {
    log("FFT size too small for exponent (%.2f bits/word).\n", bitsPerWord);
    throw "FFT size too small";
  }

  if (bitsPerWord < 1.5) {
    log("FFT size too large for exponent (%.2f bits/word).\n", bitsPerWord);
    throw "FFT size too large";
  }

  log("Expected maximum carry32: %X0000\n", config.getMaxCarry32(N, E));

  bool useLongCarry = (bitsPerWord < 10.5f) || (args.carry == Args::CARRY_LONG);

  if (useLongCarry) { log("using long carry kernels\n"); }

  bool timeKernels = args.timeKernels;

  return make_unique<Gpu>(args, E, WIDTH, SMALL_HEIGHT * MIDDLE, SMALL_HEIGHT, nW, nH,
                          getDevice(args.device), timeKernels, useLongCarry, isPm1);
}

vector<u32> Gpu::readAndCompress(ConstBuffer<int>& buf)  {
  for (int nRetry = 0; nRetry < 3; ++nRetry) {
    sum64(bufSumOut, u32(buf.size * sizeof(int)), buf);
    vector<u64> expectedVect(1);
    bufSumOut >> expectedVect;
    vector<int> data = readOut(buf);
    u64 expectedSum = expectedVect[0];
    u64 sum = 0;
    for (auto it = data.begin(), end = data.end(); it < end; it += 2) {
      sum += u32(*it) | (u64(*(it + 1)) << 32);
    }
    if (sum != expectedSum) {
      log("GPU -> Host read #%d failed (check %x vs %x)\n", nRetry, unsigned(sum), unsigned(expectedSum));
    } else {
      return compactBits(std::move(data),  E);
    }
  }
  throw "GPU -> Host persistent read errors";
}

void Gpu::tailMul(Buffer<double>& out, Buffer<double>& in, Buffer<double>& inTmp) {
  if (true) {
    tailFusedMul(out, in, inTmp);
  } else {
    fftHin(out, inTmp);
    fftHin(inTmp, in);
    multiply(out, inTmp);
    fftHout(out);
  }
}

// out := inA * inB;
void Gpu::mul(Buffer<int>& out, Buffer<int>& inA, Buffer<double>& inB, Buffer<double>& tmp1, Buffer<double>& tmp2, bool mul3) {
    fftP(tmp1, inA);
    tW(tmp2, tmp1);
    tailMul(tmp1, tmp2, inB);
    tH(tmp2, tmp1);
    fftW(tmp1, tmp2);
    if (mul3) { carryM(out, buf1); } else { carryA(out, buf1); }
    carryB(out);
}

// out := inA * inB;
void Gpu::modMul(Buffer<int>& out, Buffer<int>& inA, Buffer<int>& inB, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3, bool mul3) {

  fftP(buf1, inB);
  tW(buf3, buf1);
  
  mul(out, inA, buf3, buf1, buf2, mul3);
};

void Gpu::mul(Buffer<int>& io, Buffer<int>& inB) { modMul(io, io, inB, buf1, buf2, buf3); }

void Gpu::writeState(const vector<u32> &check, u32 blockSize, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
  assert(blockSize > 0);
  writeCheck(check);
  bufData << bufCheck;
  bufAux  << bufCheck;

  u32 n = 0;
  for (n = 1; blockSize % (2 * n) == 0; n *= 2) {
    modSqLoop(bufData, 0, n);
    modMul(bufData, bufData, bufAux, buf1, buf2, buf3);
    bufAux << bufData;
  }

  assert((n & (n - 1)) == 0);
  assert(blockSize % n == 0);
    
  blockSize /= n;
  assert(blockSize >= 2);
  
  for (u32 i = 0; i < blockSize - 2; ++i) {
    modSqLoop(bufData, 0, n);
    modMul(bufData, bufData, bufAux, buf1, buf2, buf3);
  }
  
  modSqLoop(bufData, 0, n);
  modMul(bufData, bufData, bufAux, buf1, buf2, buf3, true);
}
  
bool Gpu::doCheck(u32 blockSize, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
  modSqLoopMul3(bufAux, bufCheck, 0, blockSize);  
  modMul(bufCheck, bufCheck, bufData, buf1, buf2, buf3);  
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

void Gpu::tW(Buffer<double>& out, Buffer<double>& in) {
  fftMiddleIn(out, in);
}

void Gpu::tH(Buffer<double>& out, Buffer<double>& in) {
  fftMiddleOut(out, in);
}

void Gpu::tailMulDelta(Buffer<double>& out, Buffer<double>& in, Buffer<double>& bufA, Buffer<double>& bufB) {
  if (args.uses("NO_P2_FUSED_TAIL")) {
    fftHin(out, in);
    multiplyDelta(out, bufA, bufB);
    fftHout(out);
  } else {
    tailFusedMulDelta(out, in, bufA, bufB);
  }
}

vector<int> Gpu::readOut(ConstBuffer<int> &buf) {
  transposeOut(bufAux, buf);
  return bufAux.read();
}

void Gpu::writeIn(Buffer<int>& buf, const vector<u32>& words) { writeIn(buf, expandBits(words, N, E)); }

void Gpu::writeIn(Buffer<int>& buf, const vector<i32>& words) {
  bufAux = words;
  transposeIn(buf, bufAux);
}

// io *= in; with buffers in low position.
void Gpu::multiplyLow(Buffer<double>& io, const Buffer<double>& in, Buffer<double>& tmp) {
  // multiply(io, in); fftHout(io);
  tailMulLowLow(io, in);
  tH(tmp, io);
  carryFused(io, tmp);
  tW(tmp, io);
  fftHin(io, tmp);
}

namespace {
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

void spin() {
  static size_t spinPos = 0;
  const char spinner[] = "-\\|/";
  printf("\r%c", spinner[spinPos]);
  fflush(stdout);
  if (++spinPos >= sizeof(spinner) - 1) { spinPos = 0; }
}

}

Words Gpu::expExp2(const Words& A, u32 n) {
  u32 blockSize = 400;
  u32 logStep = 20000;
  
  writeData(A);
  IterationTimer timer{0};
  u32 k = 0;
  while (true) {
    u32 its = std::min(blockSize, n - k);
    modSqLoop(bufData, 0, its);
    k += its;
    spin();
    queue->finish();
    if (k % logStep == 0) { log("%u / %u, %.0f us/it\n", k, n, timer.reset(k) * 1'000'000.f); }
    if (k >= n) { break; }
  }
  return readData();
}

// A:= A^h * B
void Gpu::expMul(Buffer<i32>& A, u64 h, Buffer<i32>& B) {
  exponentiateHigh(A, h, buf1, buf2, buf3);
  modMul(A, A, B, buf1, buf2, buf3);
}

// return A^x * B
Words Gpu::expMul(const Words& A, u64 h, const Words& B) {
  writeData(A);
  writeCheck(B);
  expMul(bufData, h, bufCheck);
  return readData();
}

void Gpu::exponentiateHigh(Buffer<int>& bufInOut, u64 exp, Buffer<double>& bufBaseLow, Buffer<double>& buf1, Buffer<double>& buf2) {
  if (exp == 0) {
    bufInOut.set(1);
  } else if (exp > 1) {
    fftP(buf1, bufInOut);
    tW(buf2, buf1);
    fftHin(bufBaseLow, buf2);
    exponentiateCore(buf1, bufBaseLow, exp, buf2);
    fftW(buf2, buf1);
    carryA(bufInOut, buf2);
    carryB(bufInOut);
  }
}

// All buffers are in "low" position.
void Gpu::exponentiateLow(Buffer<double>& out, const Buffer<double>& base, u64 exp, Buffer<double>& tmp, Buffer<double>& tmp2) {
  assert(exp > 0);
  if (exp == 1) {
    out << base;    
  } else {
    exponentiateCore(out, base, exp, tmp);
    carryFused(tmp, out);
    tW(tmp2, tmp);
    fftHin(out, tmp2);
  }
}

static bool testBit(u64 x, int bit) { return x & (u64(1) << bit); }

// does either carrryFused() or the expanded version depending on useLongCarry
void Gpu::doCarry(Buffer<double>& out, Buffer<double>& in) {
  if (useLongCarry) {
    fftW(out, in);
    carryA(in, out);
    carryB(in);
    fftP(out, in);
  } else {
    carryFused(out, in);
  }
}

// See "left-to-right binary exponentiation" on wikipedia
void Gpu::exponentiateCore(Buffer<double>& out, const Buffer<double>& base, u64 exp, Buffer<double>& tmp) {
  assert(exp >= 2);

  tailSquareLow(tmp, base);
  tH(out, tmp);
  
  int p = 63;
  while (!testBit(exp, p)) { --p; }
  
  for (--p; ; --p) {
    if (testBit(exp, p)) {
      doCarry(tmp, out);
      tW(out, tmp);
      tailFusedMulLow(tmp, out, base);
      tH(out, tmp);
    }
    
    if (!p) { break; }

    doCarry(tmp, out);
    tW(out, tmp);
    tailSquare(tmp, out);
    tH(out, tmp);
  }
}

void Gpu::coreStep(Buffer<int>& out, Buffer<int>& in, bool leadIn, bool leadOut, bool mul3) {
  if (leadIn) { fftP(buf1, in); }
  
  tW(buf2, buf1);

  tailSquare(buf1, buf2);
  
  tH(buf2, buf1);

  if (leadOut) {
    fftW(buf1, buf2);    
    if (mul3) { carryM(out, buf1); } else { carryA(out, buf1); }    
    carryB(out);
  } else {
    if (mul3) { carryFusedMul(buf1, buf2); } else { carryFused(buf1, buf2); }
  }  
}

u32 Gpu::modSqLoop(Buffer<int>& io, u32 from, u32 to) {
  assert(from <= to);
  bool leadIn = true;
  for (u32 k = from; k < to; ++k) {
    bool leadOut = useLongCarry || (k == to - 1);
    coreStep(io, io, leadIn, leadOut, false);
    leadIn = leadOut;
  }
  return to;
}

u32 Gpu::modSqLoopMul3(Buffer<int>& out, Buffer<int>& in, u32 from, u32 to) {
  assert(from < to);
  bool leadIn = true;
  for (u32 k = from; k < to; ++k) {
    bool leadOut = useLongCarry || (k == to - 1);
    coreStep(out, (k==from) ? in : out, leadIn, leadOut, (k == to - 1));
    leadIn = leadOut;
  }
  return to;
}

bool Gpu::equalNotZero(Buffer<int>& buf1, Buffer<int>& buf2) {
  bufSmallOut.zero(1);
  u32 sizeBytes = N * sizeof(int);
  isNotZero(bufSmallOut, sizeBytes, buf1);
  isEqual(bufSmallOut, sizeBytes, buf1, buf2);
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

static string makeLogStr(u32 E, string_view status, u32 k, u64 res, float secsPerIt, u32 nIters) {
  char buf[256];
  
  snprintf(buf, sizeof(buf), "%u %2s %8d %6.2f%%; %4.0f us/it; ETA %s; %s",
           E, status.data(), k, k / float(nIters) * 100,
           secsPerIt * 1'000'000, getETA(k, nIters, secsPerIt).c_str(),
           hex(res).c_str());
  return buf;
}

static void doBigLog(u32 E, u32 k, u64 res, bool checkOK, double secsPerIt, u32 nIters, u32 nErrors) {
  log("%s%s\n", makeLogStr(E, checkOK ? "OK" : "EE", k, res, secsPerIt, nIters).c_str(),
      (nErrors ? " "s + to_string(nErrors) + " errors"s : ""s).c_str());
}

static void logPm1Stage1(u32 E, u32 k, u64 res, float secsPerIt, u32 nIters) {
  log("%s\n", makeLogStr(E, "P1", k, res, secsPerIt, nIters).c_str());
}

[[maybe_unused]] static void logPm1Stage2(u32 E, float ratioComplete) {
  char buf[256];
  snprintf(buf, sizeof(buf), "%u %2s %5.2f%%\n", E, "P2", ratioComplete * 100);
}

bool Gpu::equals9(const Words& a) {
  if (a[0] != 9) { return false; }
  for (auto it = next(a.begin()); it != a.end(); ++it) { if (*it) { return false; }}
  return true;
}

PRPState Gpu::loadPRP(u32 E, u32 iniBlockSize, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
  PRPState loaded(E, iniBlockSize);
  writeState(loaded.check, loaded.blockSize, buf1, buf2, buf3);

  u64 res64 = dataResidue();
  bool ok = (res64 == loaded.res64);

  // modMul(bufCheck, bufCheck, bufData, buf1, buf2, buf3);
  
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

static void doDiv3(u32 E, Words& words) {
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

void Gpu::doDiv9(u32 E, Words& words) {
  doDiv3(E, words);
  doDiv3(E, words);
}

namespace {
u32 checkStepForErrors(u32 argsCheckStep, u32 nErrors) {
  if (argsCheckStep) { return argsCheckStep; }  
  switch (nErrors) {
    case 0:  return 200'000;
    case 1:  return 100'000;
    default: return  50'000;
  }
}

int nFitBufs(cl_device_id device, size_t bufSize) {
  // log("availableBytes %lu\n", AllocTrac::availableBytes());
  int n = AllocTrac::availableBytes() / bufSize;  
  if (hasFreeMemInfo(device)) {
    // log("freemem %lu\n", getFreeMem(device));
    int n2 = (i64(getFreeMem(device)) - 256 * 1024 * 1024) / bufSize;
    n = std::min(n, n2);
  }
  return n;
}

template<typename To, typename From> To pun(From x) {
  static_assert(sizeof(To) == sizeof(From));
  union {
    From from;
    To to;
  } u;
  u.from = x;
  return u.to;
}

template<typename T> float asFloat(T x) { return pun<float>(x); }

}

void Gpu::printRoundoff(u32 E) {
  u32 roundN = bufRoundoff.read(1)[0];
  // fprintf(stderr, "roundN %u\n", roundN);

  vector<u32> carry;
  vector<u32> carryMul;
  bufCarryMax.readAsync(carry, 4);
  bufCarryMulMax.readAsync(carryMul, 4);
  
  vector<u32> roundVect;
  bufRoundoff.readAsync(roundVect, roundN, 8);
    
  vector<u32> zero{0, 0, 0, 0};
  bufRoundoff    = zero;
  bufCarryMax    = zero;
  bufCarryMulMax = zero;

  if (!roundN) { return; }
  if (roundN < 2000) { return; }
  
#if DUMP_STATS
  {
    File fo = File::openAppend("roundoff.txt");
    if (fo) { for (u32 x : roundVect) { fprintf(fo.get(), "%f\n", asFloat(x)); } }
  }
#endif

  double sum = 0;
  for (u32 x : roundVect) { sum += asFloat(x); }
  double avg = sum / roundN;
  double variance = 0;
  double m = 0;
  for (u32 u : roundVect) {
    double x = asFloat(u);
    m = max(m, x);
    double d = x - avg;
    variance += d * d;
  }
  
  variance /= roundN;
  double sdev = sqrt(variance);
  
  double gamma = 0.577215665; // Euler-Mascheroni
  double z = (0.5 - avg) / sdev;

  // See Gumbel distribution https://en.wikipedia.org/wiki/Gumbel_distribution
  double p = -expm1(-exp(-z * (M_PI / sqrt(6))) * (E * exp(-gamma))); 
  
  log("Roundoff: N=%u, mean %f, SD %f, CV %f, max %f, z %.1f (pErr %f%%)\n",
      roundN, avg, sdev, sdev / avg, m, z, p * 100);

#if 0
  u32 carryN = carry[3];
  u32 carryAvg = carryN ? read64(&carry[0]) / carryN : 0;
  u32 carryMax = carry[2];
  
  u32 carryMulN = carryMul[3];
  u32 carryMulAvg = carryMulN ? read64(&carryMul[0]) / carryMulN : 0;
  u32 carryMulMax = carryMul[2];

  log("Carry: N=%u, max %x, avg %x; CarryM: N=%u, max %x, avg %x\n",
      carryN, carryMax, carryAvg, carryMulN, carryMulMax, carryMulAvg);
#endif
}

void Gpu::accumulate(Buffer<int>& acc, Buffer<double>& data, Buffer<double>& tmp1, Buffer<double>& tmp2) {
  fftP(tmp1, acc);
  tW(tmp2, tmp1);
  tailMul(tmp1, data, tmp2);
  tH(tmp2, tmp1);
  fftW(tmp1, tmp2);
  carryA(acc, tmp1);
  carryB(acc);
}

void Gpu::square(Buffer<int>& data, Buffer<double>& tmp1, Buffer<double>& tmp2) {
  fftP(tmp1, data);
  tW(tmp2, tmp1);
  tailSquare(tmp1, tmp2);
  tH(tmp2, tmp1);
  fftW(tmp1, tmp2);
  carryA(data, tmp1);
  carryB(data);  
}

class B1Accumulator {
public:
  Gpu *gpu;
  const u32 E;
  const u32 b1;
  
  u32 nextK = 0;

  vector<bool> bits;
  vector<Buffer<i32>> bufs;
  vector<bool> engaged;

  B1Accumulator(Gpu* gpu, u32 E, u32 b1) : gpu{gpu}, E{E}, b1{b1} {}

  void set(u32 k, B1State& state, u32 nBufs) {
    assert(nBufs > 0);
    engaged.clear();
    
    if (!b1) {
      assert(bufs.empty());
      assert(nextK == 0);
      state.data.clear();
      return;
    }

    if (k == 0) {
      assert(state.isEmpty() && state.data.empty());
    } else {    
      if (b1 != state.b1) {
        log("B1 requested %u but savefile has %u\n", b1, state.b1);
        throw("mismatched B1");
      }

      if (state.isCompleted()) {
        assert(state.data.empty());
        assert(bufs.empty());
        nextK = 0;
        log("(B1=%u completed)\n", state.b1);
        return;
      }      
    }

    while (bufs.size() > nBufs) { bufs.pop_back(); }
    if (bufs.size() < nBufs) {
      auto tail = gpu->makeBufVector(nBufs - bufs.size());
      std::move(tail.begin(), tail.end(), std::back_inserter(bufs));      
    }

    engaged.resize(nBufs);
    assert(bufs.size() == nBufs && engaged.size() == nBufs);

    if (bits.empty()) {
      Timer timer;
      bits = powerSmoothLSB(E, b1);
      log("GMP powerSmooth(%u) took %.2fs\n", b1, timer.elapsedSecs());
    }
    
    if (k == 0) {
      nextK = 0;
      while (nextK < bits.size() && !bits[nextK]) { ++nextK; }
      assert(nextK < bits.size());
    } else {
      assert(!state.data.empty());
      assert(bits.size() == state.nBits);
      
      gpu->writeIn(bufs[0], state.data);
      state.data.clear();
      engaged[0] = true;
      nextK = state.nextK;
      assert(nextK > k && nextK < bits.size());
      assert(bits[nextK]);
    }
  }
  

  bool isComplete() { return !nextK; }

  void step(u32 kAt, Buffer<int>& data) {
    if (kAt != nextK) { return; }
    
    assert(nextK < bits.size() && bits[nextK]);
    
    u32 start = nextK + 1;
    u32 sum = 0;
    u32 i = start;
    
    for (; i < bits.size(); ++i) {
      if (bits[i]) {
        u32 delta = 1u << (i - start);
        if (i - start > 31 || sum + delta >= bufs.size()) {
          break;
        } else {
          sum += delta;
        }
      }
    }

    nextK = i;
    assert(nextK == bits.size() || bits[nextK]);

    if (nextK == bits.size()) { nextK = 0; }

    assert(sum < bufs.size());
    if (engaged[sum]) {
      gpu->mul(bufs[sum], data);
    } else {
      bufs[sum] << data;
      engaged[sum] = true;
    }
  }

  vector<u32> fold(Buffer<double>& tmp1, Buffer<double>& tmp2, Buffer<double>& tmp3) {
    u32 n = bufs.size();
    assert(n > 1);

    Timer timer;
    for (int i = n-2; i > 0; --i) {
      gpu->modMul(bufs[i],   bufs[i], bufs[i+1], tmp1, tmp2, tmp3);
      gpu->modMul(bufs[n-1], bufs[n-1], bufs[i],   tmp1, tmp2, tmp3);
    }
    gpu->square(bufs[n-1], tmp1, tmp2);
    gpu->modMul(bufs[0], bufs[0], bufs[1], tmp1, tmp2, tmp3);    
    gpu->modMul(bufs[0], bufs[0], bufs[n-1], tmp1, tmp2, tmp3);
    gpu->finish();
    log("B1 fold(%u) took %.2fs\n", n, timer.deltaSecs());
    
    for (u32 i = 1; i < n; ++i) { bufs[i].set(1); }
    gpu->finish();
    log("B1 reset(%u) took %.2fs\n", n, timer.deltaSecs());

    auto ret = gpu->readAndCompress(bufs[0]);
    log("B1 read took %.2fs\n", timer.deltaSecs());
    return ret;
  }
};

tuple<bool, u64, u32, string> Gpu::isPrimePRP(u32 E, const Args &args, std::atomic<u32>& factorFoundForExp, u32 b1, u32 b1Low) {
  u32 k = 0, blockSize = 0, nErrors = 0;

  if (!args.maxAlloc) { log("Use -maxAlloc <MBytes> to bound GPU memory usage\n"); }

  u32 power = -1;
  u32 startK = 0;

  const u32 b1Bufs = [this]() {
    u32 smallBufSize = N * 4; // "small buf" of ints
    size_t availableBytes = AllocTrac::availableBytes();
    u32 maxBufs = (availableBytes + smallBufSize / 2) / smallBufSize;
    assert(maxBufs >= 13);     // An arbitrary too-low-memory point.
    u32 b1Bufs = maxBufs - 5;  // Keep a small reserve of free RAM.
    float MB = 1 / (1024.0f*1024.0f);
    log("Space for %u B1 buffers (available mem %.1f MB, buf size %.1fMB)\n",  b1Bufs, availableBytes * MB, smallBufSize * MB);
    return b1Bufs; }();

  B1Accumulator lowAcc{this, E, b1Low};
  B1Accumulator highAcc{this, E, b1};

 reload:
  {
    PRPState loaded(E, args.blockSize);

    B1State& lowB1  = loaded.lowB1;
    B1State& highB1 = loaded.highB1;


    if (b1 && highB1.b1 != b1) {
      log("Requested high B1=%u but savefile has B1=%u\n", b1, highB1.b1);
      throw "B1 mismatch";
    }
    
    if (b1Low && lowB1.b1 != b1Low) {
      log("Requested low B1=%u but savefile has B1=%u\n", b1Low, lowB1.b1);
      throw "B1 mismatch";
    }

    writeState(loaded.check, loaded.blockSize, buf1, buf2, buf3);
    
    u64 res64 = dataResidue();
    bool ok = (res64 == loaded.res64);

    std::string expected = " (expected "s + hex(loaded.res64) + ")";
    log("%u %2s %8d loaded: blockSize %d, %s%s\n",
        E, ok ? "OK" : "EE", loaded.k, loaded.blockSize, hex(res64).c_str(), ok ? "" : expected.c_str());
    
    if (!ok) { throw "error on load"; }

    k = loaded.k;
    blockSize = loaded.blockSize;
    if (nErrors == 0) { nErrors = loaded.nErrors; }
    assert(nErrors >= loaded.nErrors);


    /*
    if (lowB1.b1) {
      assert(k <= lowB1.nextK);
      assert(highB1.b1 > lowB1.b1);
      assert(k <= highB1.nextK);
      b1Bufs /= 2; // split equally between lowB1 and highB1.
    }
    */
    
    lowAcc.set(k, lowB1, b1Bufs/2);
    highAcc.set(k, highB1, b1Bufs - lowAcc.bufs.size());
  }

  assert(blockSize > 0 && 10000 % blockSize == 0);
  
  u32 checkStep = checkStepForErrors(args.logStep, nErrors);

  if (!startK) { startK = k; }

  if (power == u32(-1)) {
    power = ProofSet::effectivePower(args.tmpDir, E, args.proofPow, startK);  
    if (!power) {
      log("Proof disabled because of missing checkpoints\n");
    } else if (power != args.proofPow) {
      log("Proof using power %u (vs %u) for %u\n", power, args.proofPow, E);
    } else {
      log("Proof using power %u\n", power);
    }
  }
  
  ProofSet proofSet{args.tmpDir, E, power};

  Signal signal;

  // Number of sequential errors (with no success in between). If this ever gets high enough, stop.
  int nSeqErrors = 0;

  bool isPrime = false;
  IterationTimer itTimer{startK};

  u64 finalRes64 = 0;

  // We extract the res64 at kEnd.
  const u32 kEnd = E; // Type-1 per http://www.mersenneforum.org/showpost.php?p=468378&postcount=209
  assert(k < kEnd);

  // We continue beyound kEnd: up to the next multiple of 1024 if proof is enabled (kProofEnd), and up to the next blockSize
  u32 kEndEnd = roundUp(proofSet.kProofEnd(kEnd), blockSize);

  bool displayRoundoff = args.flags.count("STATS");

  bool skipNextCheckUpdate = false;

  u32 persistK = proofSet.firstPersistAt(k + 1);
  bool leadIn = true;  

  assert(k % blockSize == 0);
  assert(checkStep % blockSize == 0);
  
  while (true) {
    assert(k < kEndEnd);   

    if (skipNextCheckUpdate) {
      skipNextCheckUpdate = false;
    } else if (k % blockSize == 0) {
      modMul(bufCheck, bufCheck, bufData, buf1, buf2, buf3);
    }

    u32 nextK = k + 1;
    bool leadOut = (nextK % blockSize == 0) || nextK == persistK || nextK == kEnd || nextK == highAcc.nextK || nextK == lowAcc.nextK;

    coreStep(bufData, bufData, leadIn, leadOut, false);
    leadIn = leadOut;
    ++k;
    
    if (k == persistK) {
      proofSet.save(k, readData());
      persistK = proofSet.firstPersistAt(k + 1);
    }

    if (k == kEnd) {
      auto words = readData();
      isPrime = equals9(words);
      doDiv9(E, words);
      finalRes64 = residue(words);
      log("%s %8d / %d, %s\n", isPrime ? "PP" : "CC", kEnd, E, hex(finalRes64).c_str());
    }

    lowAcc.step(k, bufData);
    highAcc.step(k, bufData);
    
    if (k % blockSize == 0) {
      if (!args.noSpin) { spin(); }

      queue->finish();

      if (factorFoundForExp == E) {
        log("Aborting the PRP test because a factor was found\n");
        factorFoundForExp = 0;
        return {false, 0, u32(-1), {}};
      }

      bool doStop = signal.stopRequested() || (args.iters && k - startK == args.iters);
      if (doStop) {
        log("Stopping, please wait..\n");
        signal.release();
      }

      bool doCheck = doStop || (k % checkStep == 0) || (k >= kEndEnd) || (k - startK == 2 * blockSize);

      if (doCheck) {
        if (displayRoundoff) { printRoundoff(E); }
      
        u64 res64 = dataResidue();
        PRPState prpState{E, k, blockSize, res64, readCheck(), nErrors};
        bool ok = this->doCheck(blockSize, buf1, buf2, buf3);
      
        if (ok) {
          skipNextCheckUpdate = true;
          double timeWithoutSave = itTimer.reset(k);
          if (k < kEnd) { prpState.save(false); }
          doBigLog(E, k, res64, ok, timeWithoutSave, kEndEnd, nErrors);
          if (k >= kEndEnd) {
            fs::path proofPath;
            if (proofSet.power > 0) {
              proofPath = proofSet.computeProof(this).save(args.proofResultDir);
              log("PRP-Proof '%s' generated\n", proofPath.string().c_str());
              if (!args.keepProof) {
                log("Proof: cleaning up temporary storage\n");
                proofSet.cleanup();
              }
            }
            return {isPrime, finalRes64, nErrors, proofPath.string()};
          }
          nSeqErrors = 0;      
        } else {
          doBigLog(E, k, res64, ok, itTimer.reset(k), kEndEnd, nErrors);
          ++nErrors;
          if (++nSeqErrors > 2) {
            log("%d sequential errors, will stop.\n", nSeqErrors);
            throw "too many errors";
          }
          goto reload;
        }
        
        logTimeKernels();
        if (doStop) { throw "stop requested"; }
        itTimer.reset(k);
      }
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
  
  SquaringSet(Gpu& gpu, u32 N, const Buffer<double>& bufBase, Buffer<double>& bufTmp, Buffer<double>& bufTmp2, array<u64, 3> exponents, string_view name)
    : SquaringSet(gpu, N, name) {
    
    gpu.exponentiateLow(C, bufBase, exponents[0], bufTmp, bufTmp2);
    gpu.exponentiateLow(B, bufBase, exponents[1], bufTmp, bufTmp2);
    if (exponents[2] == exponents[1]) {
      A << B;
    } else {
      gpu.exponentiateLow(A, bufBase, exponents[2], bufTmp, bufTmp2);
    }
  }

  SquaringSet& operator=(const SquaringSet& rhs) {
    assert(N == rhs.N);
    copyFrom(rhs);
    return *this;
  }

  void step(Buffer<double>& bufTmp) {
    gpu.multiplyLow(C, B, bufTmp);
    gpu.multiplyLow(B, A, bufTmp);
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
    coreStep(bufData, bufData, leadIn, leadOut, bits[k]);
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
  if (leadIn) { fftP(buf2, bufData); }

  HostAccessBuffer<double> bufAcc{queue, "acc", N};

  tW(buf1, buf2);
  tailSquare(buf2, buf1);
  tH(bufAcc, buf2);			// Save bufAcc for later use as an accumulator
  fftW(buf2, bufAcc);
  carryA(bufData, buf2);
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
  fftP(bufBase, bufData);
  tW(buf1, bufBase);
  fftHin(bufBase, buf1);
  
  auto [startBlock, nPrimes, allSelected] = makePm1Plan(B1, B2);
  assert(startBlock > 0);  
  u32 nBlocks = allSelected.size();
  log("%u P2 using blocks [%u - %u] to cover %u primes\n", E, startBlock, startBlock + nBlocks - 1, nPrimes);
  
  exponentiateLow(buf2, bufBase, 30030*30030, buf1, buf3); // Aux := base^(D^2)

  constexpr auto jset = getJset();
  static_assert(jset[0] == 1);
  static_assert(jset[2880 - 1] == 15013);

  u32 beginJ = jset[beginPos];
  SquaringSet little{*this, N, bufBase, buf1, buf3, {beginJ*beginJ, 4 * (beginJ + 1), 8}, "little"};
  SquaringSet bigStart{*this, N, buf2, buf1, buf3, {u64(startBlock)*startBlock, 2 * startBlock + 1, 2}, "bigStart"};
  bufBase.reset();
  buf2.reset();
  SquaringSet big{*this, N, "big"};
  

  vector<Buffer<double>> blockBufs;
  try {
    for (int i = 0, end = std::min(2880/2, nFitBufs(device, sizeof(double) * N)); i < end; ++i) {
      blockBufs.emplace_back(queue, "p2BlockBuf"s + std::to_string(i), N);
    }
  } catch (const gpu_bad_alloc& e) {
  }

  vector<u32> stage2Data;
  
  if (blockBufs.size() < 4) {
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
      for (int steps = delta / 2; steps > 0; --steps) { little.step(buf1); }
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
        big.step(buf1);
      }
      for (u32 i = 0; i < nUsedBufs; ++i) {
        if (selected[pos + i]) {
          ++nSelected;
          carryFused(buf1, bufAcc);
          tW(bufAcc, buf1);
          tailMulDelta(buf1, bufAcc, big.C, blockBufs[i]);
          tH(bufAcc, buf1);
        }
      }
      queue->finish();
      if (gcdFuture.valid() && gcdFuture.wait_for(chrono::steady_clock::duration::zero()) == future_status::ready) {
        string gcd = gcdFuture.get();
        log("%u P1 GCD: %s\n", E, gcd.empty() ? "no factor" : gcd.c_str());
        if (!gcd.empty()) { return gcd; }
      }
    }

    if (pos + nBufs < 2880) { P2State{E, B1, B2, pos + nBufs, bufAcc.read()}.save(); }
    log("%u P2 %4u/2880: %u primes; setup %5.2f s, %7.3f ms/prime\n", E, pos + nUsedBufs, nSelected, setup, timer.deltaSecs() * 1000.f / (nSelected + 1));
    logTimeKernels();
  }

  fftW(buf1, bufAcc);
  carryA(bufData, buf1);
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

// Copyright Mihai Preda and George Woltman.

#include "Gpu.h"
#include "Proof.h"
#include "Pm1Plan.h"
#include "Saver.h"
#include "state.h"
#include "Args.h"
#include "Signal.h"
#include "FFTConfig.h"
#include "GmpUtil.h"
#include "AllocTrac.h"
#include "Queue.h"
#include "Task.h"
#include "Memlock.h"
#include "B1Accumulator.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <algorithm>
#include <future>
#include <optional>
#include <numeric>
#include <bitset>
#include <limits>
#include <iomanip>
#include <bit>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static_assert(sizeof(double2) == 16, "size double2");
static_assert(sizeof(long double) > sizeof(double), "long double offers extended precision");

extern const char *CL_SOURCE;

struct Weights {
  vector<float> threadWeightsIF;
  vector<float> carryWeightsIF;
  vector<u32> bitsCF;
  vector<u32> bitsC;
};

namespace {

// Returns the primitive root of unity of order N, to the power k.

template<typename T>
pair<T, T> root1(u32 N, u32 k) {
  assert(k < N);
  if (k >= N/2) {
    auto [c, s] = root1<T>(N, k - N/2);
    return {-c, -s};
  } else if (k > N/4) {
    auto [c, s] = root1<T>(N, N/2 - k);
    return {-c, s};
  } else if (k > N/8) {
    auto [c, s] = root1<T>(N, N/4 - k);
    return {-s, -c};
  } else {
    assert(!(N&7));
    assert(k <= N/8);
    N /= 2;
    auto angle = - M_PIl * k / N;
    return {cosl(angle), sinl(angle)};
  }
}

template<typename T>
pair<T, T> *smallTrigBlock(u32 W, u32 H, pair<T, T> *p) {
  for (u32 line = 1; line < H; ++line) {
    for (u32 col = 0; col < W; ++col) {
      *p++ = root1<T>(W * H, line * col);
    }
  }
  return p;
}

template<typename T>
ConstBuffer<pair<T, T>> genSmallTrig(const Context& context, u32 size, u32 radix) {
  vector<pair<T, T>> tab;

  for (u32 line = 1; line < radix; ++line) {
    for (u32 col = 0; col < size / radix; ++col) {
      tab.push_back(root1<T>(size, col * line));
    }
  }
  tab.resize(size);
  return {context, "smallTrig", tab};
}

template<typename T>
ConstBuffer<pair<T, T>> genMiddleTrig(const Context& context, u32 smallH, u32 middle) {
  vector<pair<T, T>> tab;
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

template<typename T>
vector<pair<T, T>> makeTrig(u32 n) {
  assert(n % 8 == 0);
  vector<pair<T, T>> tab;
  tab.reserve(n/8 + 1);
  for (u32 k = 0; k <= n/8; ++k) { tab.push_back(root1<T>(n, k)); }
  return tab;
}

template<typename T>
vector<pair<T, T>> makeTinyTrig(u32 W, u32 hN) {
  vector<pair<T, T>> tab;
  for (u32 k = 0; k <= W/2; ++k) {
    auto[c, s] = root1<long double>(hN, k);
    tab.push_back({c - 1, s});
  }
  return tab;
}

u32 kAt(u32 H, u32 line, u32 col) { return (line + col * H) * 2; }

auto weight(u32 N, u32 E, u32 H, u32 line, u32 col, u32 rep) {
  auto iN = 1 / (f128) N;
  return exp2l(iN * extra(N, E, kAt(H, line, col) + rep));
}

auto invWeight(u32 N, u32 E, u32 H, u32 line, u32 col, u32 rep) {
  auto iN = 1 / (f128) N;
  return exp2l(- iN * extra(N, E, kAt(H, line, col) + rep));
}

// double boundUnderOne(double x) { return std::min(x, nexttoward(1, 0)); }
float boundUnderOne(float x) { return std::min(x, nexttowardf(1, 0)); }

#define CARRY_LEN 8

template<typename T>
Weights genWeights(u32 E, u32 W, u32 H, u32 nW) {
  u32 N = 2u * W * H;
  
  u32 groupWidth = W / nW;

  // Inverse + Forward
  vector<T> threadWeightsIF;
  for (u32 thread = 0; thread < groupWidth; ++thread) {
    auto iw = invWeight(N, E, H, 0, thread, 0);
    threadWeightsIF.push_back(iw - 1);
    auto w = weight(N, E, H, 0, thread, 0);
    threadWeightsIF.push_back(w - 1);
  }

  // the group order matches CarryA/M (not fftP/CarryFused).
  vector<T> carryWeightsIF;
  for (u32 gy = 0; gy < H / CARRY_LEN; ++gy) {
    auto iw = invWeight(N, E, H, gy * CARRY_LEN, 0, 0);
    carryWeightsIF.push_back(2 * boundUnderOne(T(iw)));
    
    auto w = weight(N, E, H, gy * CARRY_LEN, 0, 0);
    carryWeightsIF.push_back(2 * w);
  }
  
  vector<u32> bits;
  
  for (u32 line = 0; line < H; ++line) {
    for (u32 thread = 0; thread < groupWidth; ) {
      std::bitset<32> b;
      for (u32 bitoffset = 0; bitoffset < 32; bitoffset += nW*2, ++thread) {
        for (u32 block = 0; block < nW; ++block) {
          for (u32 rep = 0; rep < 2; ++rep) {
            if (isBigWord(N, E, kAt(H, line, block * groupWidth + thread) + rep)) { b.set(bitoffset + block * 2 + rep); }
          }        
	}
      }
      bits.push_back(b.to_ulong());
    }
  }
  assert(bits.size() == N / 32);
  
  vector<u32> bitsC;
  
  for (u32 gy = 0; gy < H / CARRY_LEN; ++gy) {
      for (u32 thread = 0; thread < W; ) {
        std::bitset<32> b;
        for (u32 bitoffset = 0; bitoffset < 32; bitoffset += CARRY_LEN * 2, ++thread) {
          for (u32 block = 0; block < CARRY_LEN; ++block) {
            for (u32 rep = 0; rep < 2; ++rep) {
              if (isBigWord(N, E, kAt(H, gy * CARRY_LEN + block, thread) + rep)) { b.set(bitoffset + block * 2 + rep); }
            }
          }
        }
        bitsC.push_back(b.to_ulong());
      }
  }
  assert(bitsC.size() == N / 32);

  return Weights{threadWeightsIF, carryWeightsIF, bits, bitsC};
}

string toLiteral(u32 value) { return to_string(value) + 'u'; }
string toLiteral(i32 value) { return to_string(value); }
[[maybe_unused]] string toLiteral(u64 value) { return to_string(value) + "ul"; }

template<typename F>
string toLiteral(F value) {
  std::ostringstream ss;
  ss << std::setprecision(numeric_limits<F>::max_digits10) << value;
  string s = std::move(ss).str();

  // verify exact roundtrip
  [[maybe_unused]] F back = 0;
  sscanf(s.c_str(), (sizeof(F) == 4) ? "%f" : "%lf", &back);
  assert(back == value);
  
  return s;
}

template<typename T>
string toLiteral(const vector<T>& v) {
  assert(!v.empty());
  string s = "{";
  for (auto x : v) {
    s += toLiteral(x) + ",";
  }
  s += "}";
  return s;
}

struct Define {
  const string str;

  template<typename T> Define(const string& label, T value) : str{label + '=' + toLiteral(value)} {
    assert(label.find('=') == string::npos);
  }

  explicit Define(const string& labelAndVal) : str{labelAndVal} {
    assert(labelAndVal.find('=') != string::npos);
  }
  
  operator string() const { return str; }
};

cl_program compile(const Args& args, cl_context context, cl_device_id id, u32 N, u32 E, u32 WIDTH, u32 SMALL_HEIGHT, u32 MIDDLE, u32 nW) {
  string clArgs = args.dump.empty() ? ""s : (" -save-temps="s + args.dump + "/" + numberK(N));
  if (!args.safeMath) { clArgs += " -cl-unsafe-math-optimizations"; }
  
  vector<Define> defines =
    {{"EXP", E},
     {"WIDTH", WIDTH},
     {"SMALL_HEIGHT", SMALL_HEIGHT},
     {"MIDDLE", MIDDLE},
    };

  if (isAmdGpu(id)) { defines.push_back({"AMDGPU", 1}); }

  defines.push_back({"WEIGHT_STEP", float(weight(N, E, SMALL_HEIGHT * MIDDLE, 0, 0, 1) - 1)});
  defines.push_back({"IWEIGHT_STEP", float(invWeight(N, E, SMALL_HEIGHT * MIDDLE, 0, 0, 1) - 1)});

  vector<float> iWeights;
  vector<float> fWeights;
  for (u32 i = 0; i < CARRY_LEN; ++i) {
    iWeights.push_back(invWeight(N, E, SMALL_HEIGHT * MIDDLE, 0, 0, 2*i) - 1);
    fWeights.push_back(weight(N, E, SMALL_HEIGHT * MIDDLE, 0, 0, 2*i) - 1);
  }
  defines.push_back({"IWEIGHTS", iWeights});
  defines.push_back({"FWEIGHTS", fWeights});
  
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
      defines.push_back(Define{flag});
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
         cl_device_id device, bool timeKernels, bool useLongCarry)
  : Gpu{args, E, W, BIG_H, SMALL_H, nW, nH, device, timeKernels, useLongCarry, genWeights<float>(E, W, BIG_H, nW)}
{}

using float2 = pair<float, float>;

Gpu::Gpu(const Args& args, u32 E, u32 W, u32 BIG_H, u32 SMALL_H, u32 nW, u32 nH,
         cl_device_id device, bool timeKernels, bool useLongCarry, Weights&& weights) :
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
  program(compile(args, context.get(), device, N, E, W, SMALL_H, BIG_H / SMALL_H, nW)),
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
  LOAD_WS(carryA,  hN / CARRY_LEN),
  LOAD_WS(carryM,  hN / CARRY_LEN),
  LOAD_WS(carryB,  hN / CARRY_LEN),
  LOAD(transposeW,   (W/64) * (BIG_H/64)),
  LOAD(transposeH,   (W/64) * (BIG_H/64)),
  LOAD(transposeIn,  (W/64) * (BIG_H/64)),
  LOAD(transposeOut, (W/64) * (BIG_H/64)),
  LOAD(kernelMultiply,      hN / SMALL_H),
  LOAD(kernelMultiplyDelta, hN / SMALL_H),
  LOAD(tailFusedSquare,   hN / SMALL_H / 2),
  LOAD(tailFusedMulDelta, hN / SMALL_H / 2),
  LOAD(tailFusedMulLow,   hN / SMALL_H / 2),
  LOAD(tailFusedMul,      hN / SMALL_H / 2),
  LOAD(tailSquareLow,     hN / SMALL_H / 2),
  LOAD(tailMulLowLow,     hN / SMALL_H / 2),
  LOAD(readResidue, 1),
  LOAD_WS(differ, hN / CARRY_LEN),
  LOAD(sum64, 256),
  LOAD(readROE, 16),
  LOAD_WS(stats, hN / 16),
#undef LOAD_WS
#undef LOAD

  bufTrigW{genSmallTrig<float>(context, W, nW)},
  bufTrigH{genSmallTrig<float>(context, SMALL_H, nH)},
  bufTrigM{genMiddleTrig<float>(context, SMALL_H, BIG_H / SMALL_H)},
  bufBits{context, "bits", weights.bitsCF},
  bufBitsC{context, "bitsC", weights.bitsC},
  bufData{queue, "data", N},
  bufAux{queue, "aux", N},
  bufTranspose{queue, "transpose", N},
  bufCheck{queue, "check", N},
  bufCarry{queue, "carry", N / 2},
  bufReady{queue, "ready", BIG_H},
  bufRoundoff{queue, "roundoff", 1024 + 1},
  bufCarryMax{queue, "carryMax", 8},
  bufCarryMulMax{queue, "carryMulMax", 8},
  bufSmallOut{queue, "smallOut", 256},
  bufZero{queue, "zero", 1},
  bufSumOut{queue, "sumOut", 1},
  bufStatsOut{queue, "stats", 3},
  buf1{queue, "buf1", N},
  buf2{queue, "buf2", N},
  buf3{queue, "buf3", N},
  args{args}
{
  // dumpBinary(program.get(), "isa.bin");
  
  carryFused.setFixedArgs(   2, bufCarry, bufReady, bufTrigW, bufBits);
  carryFusedMul.setFixedArgs(2, bufCarry, bufReady, bufTrigW, bufBits);
  fftP.setFixedArgs(2, bufTrigW);
  fftW.setFixedArgs(2, bufTrigW);
  fftHin.setFixedArgs(2, bufTrigH);
  fftHout.setFixedArgs(1, bufTrigH);
    
  carryA.setFixedArgs(2, bufCarry, bufBitsC);
  carryM.setFixedArgs(2, bufCarry, bufBitsC);
  carryB.setFixedArgs(1, bufCarry, bufBitsC);

  tailFusedMulDelta.setFixedArgs(4, bufTrigH, bufTrigH);
  tailFusedMulLow.setFixedArgs(3, bufTrigH, bufTrigH);
  tailFusedMul.setFixedArgs(3, bufTrigH, bufTrigH);
  tailMulLowLow.setFixedArgs(2, bufTrigH);
  
  tailFusedSquare.setFixedArgs(2, bufTrigH, bufTrigH);
  tailSquareLow.setFixedArgs(2, bufTrigH, bufTrigH);
  readROE.setFixedArgs(0, bufRoundoff);

  bufReady.zero();
  // bufRoundoff.zero();
  bufCarryMax.zero();
  bufCarryMulMax.zero();
  bufZero.zero();

  vector<float2> readTrigSH, readTrigBH, readTrigN;
  {
    HostAccessBuffer<float2>
      bufSH{queue, "readTrig", SMALL_H/4 + 1},
      bufBH{queue, "readTrigBH", BIG_H/8 + 1},
      bufN{queue, "readTrigN", hN/8+1};
/*
    Kernel{program.get(), queue, device, 32, "readHwTrig"}(bufSH, bufBH, bufN);
    readTrigSH = bufSH.read();
    readTrigBH = bufBH.read();
    readTrigN = bufN.read();
*/

    Kernel{program.get(), queue, device, 32, "writeGlobals"}(ConstBuffer{context, "dp1", makeTrig<float>(2 * SMALL_H)},
                                                             ConstBuffer{context, "dp2", makeTrig<float>(BIG_H)},
                                                             ConstBuffer{context, "dp3", makeTrig<float>(hN)},
                                                             ConstBuffer{context, "dp4", makeTinyTrig<float>(W, hN)},

                                                             ConstBuffer{context, "w2", weights.threadWeightsIF},
                                                             ConstBuffer{context, "w3", weights.carryWeightsIF}
                                                             );
  }

  readROE(); // zero-out ROE_MAX

  finish();
  
  program.reset();
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

Words Gpu::fold(vector<Buffer<int>>& bufs) {
  assert(!bufs.empty());

  for (int retry = 0; retry < 2; ++retry) {
    {
      Buffer<float> A{queue, "A", N};
      Buffer<float> B{queue, "B", N};
      {
        Buffer<int>& last = bufs.back();
        fftP(buf3, last);
        tW(B, buf3);
        fftHin(A, B);
      }

      u32 n = bufs.size();
      for (int i = n - 2; i >= 0; --i) {
        Buffer<int>& buf = bufs[i];
        fftP(buf3, buf);
        tW(buf2, buf3);
        tailFusedMulLow(A, buf2, A);
        tH(buf3, A);
        doCarry(A, buf3);
        tW(buf3, A);
        fftHin(A, buf3);

        if (i == 0) {
          tailSquare(buf3, B);
          tH(B, buf3);
          doCarry(buf3, B);
          tW(B, buf3);
        }
    
        tailFusedMulLow(buf3, B, A);
        tH(B, buf3);
    
        if (i == 0) {
          fftW(buf3, B);          
          break;
        } else {
          doCarry(buf3, B);
          tW(B, buf3);
        }
      }
    }

    Buffer<int> C{queue, "C", N};
    carryA(C, buf3);
    carryB(C);
    Words folded = readAndCompress(C);
    if (!folded.empty()) { return folded; }
    log("P1 fold() error ZERO, will retry\n");
  }
  return {};
}

unique_ptr<Gpu> Gpu::make(u32 E, const Args &args) {
  FFTConfig config = getFFTConfig(E, args.fftSpec);
  u32 WIDTH        = config.width;
  u32 SMALL_HEIGHT = config.height;
  u32 MIDDLE       = config.middle;
  u32 N = WIDTH * SMALL_HEIGHT * MIDDLE * 2;

  u32 nW = (WIDTH == 1024 || WIDTH == 256) ? 4 : 8;
  u32 nH = (SMALL_HEIGHT == 1024 || SMALL_HEIGHT == 256) ? 4 : 8;

  float bitsPerWord = E / float(N);
  log("FFT: %s %s (%.2f bpw)\n", numberK(N).c_str(), config.spec().c_str(), bitsPerWord);

  if (bitsPerWord > 20) {
    log("FFT size too small for exponent (%.2f bits/word).\n", bitsPerWord);
    throw "FFT size too small";
  }

  if (bitsPerWord < FFTConfig::MIN_BPW) {
    log("FFT size too large for exponent (%.2f bits/word < %.2f bits/word).\n", bitsPerWord, FFTConfig::MIN_BPW);
    throw "FFT size too large";
  }

  // log("Expected maximum carry32: %X0000\n", config.getMaxCarry32(N, E));

  bool useLongCarry = (bitsPerWord < 10.5f) || (args.carry == Args::CARRY_LONG);

  if (useLongCarry) { log("using long carry kernels\n"); }

  bool timeKernels = args.timeKernels;

  return make_unique<Gpu>(args, E, WIDTH, SMALL_HEIGHT * MIDDLE, SMALL_HEIGHT, nW, nH,
                          getDevice(args.device), timeKernels, useLongCarry);
}

vector<u32> Gpu::readAndCompress(Buffer<int>& buf, u32 *outSum)  {
  for (int nRetry = 0; nRetry < 2; ++nRetry) {
    bufStatsOut.zero();
    stats(bufStatsOut, buf);
    vector<i64> readStats(3);
    bufStatsOut.readAsync(readStats);
    vector<i32> data = readOut(buf);

    i64 sumAbs = readStats[0];
    // i64 sum = readStats[1];
    u32 sumM31 = modM31(readStats[2]);

    if (sumAbs == 0) {
      log("GPU->Host read ZERO\n");
      return {};
    }

    vector<u32> compacted = compactBits(std::move(data),  E);
    u32 c2 = modM31(compacted);
    assert(c2 == modM31(N, E, data));

    if (c2 == sumM31) {
      if (outSum) { *outSum = c2; }
      return compacted;
    }
    log("GPU->Host read checksum: expected %08x actual %08x\n", sumM31, c2);
  }
  throw "Persistent read errors: GPU->Host";
}

vector<u32> Gpu::readCheck() { return readAndCompress(bufCheck); }
vector<u32> Gpu::readData() { return readAndCompress(bufData); }

void Gpu::tailMul(TBuf& out, TBuf& in, TBuf& inTmp) {
  if (true) {
    tailFusedMul(out, in, inTmp);
  } else {
    fftHin(out, inTmp);
    fftHin(inTmp, in);
    kernelMultiply(out, inTmp);
    fftHout(out);
  }
}

// out := inA * inB;
void Gpu::mul(Buffer<int>& out, Buffer<int>& inA, TBuf& inB, TBuf& tmp1, TBuf& tmp2, bool mul3) {
    fftP(tmp1, inA);
    tW(tmp2, tmp1);
    tailMul(tmp1, inB, tmp2);
    tH(tmp2, tmp1);
    fftW(tmp1, tmp2);
    if (mul3) { carryM(out, tmp1); } else { carryA(out, tmp1); }
    carryB(out);
}

// out := inA * inB;
void Gpu::modMul(Buffer<int>& out, Buffer<int>& inA, Buffer<int>& inB, TBuf& buf1, TBuf& buf2, TBuf& buf3, bool mul3) {

  fftP(buf1, inB);
  tW(buf3, buf1);
  
  mul(out, inA, buf3, buf1, buf2, mul3);
};

void Gpu::mul(Buffer<int>& io, Buffer<int>& inA, Buffer<int>& inB) { modMul(io, inA, inB, buf1, buf2, buf3); }

void Gpu::mul(Buffer<int>& io, Buffer<int>& inB) { mul(io, io, inB); }

void Gpu::mul(Buffer<int>& io, TBuf& buf1) {
  // We know that coreStep() stores double output in buf1; so we're going to use buf2 & buf3 for temps.
  // tW(buf2, buf1);
  mul(io, io, buf1, buf2, buf3, false);
}

void Gpu::writeState(const vector<u32> &check, u32 blockSize, TBuf& buf1, TBuf& buf2, TBuf& buf3) {
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
  
bool Gpu::doCheck(u32 blockSize, TBuf& buf1, TBuf& buf2, TBuf& buf3) {
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

void Gpu::tW(TBuf& out, TBuf& in) {
  fftMiddleIn(out, in);
}

void Gpu::tH(TBuf& out, TBuf& in) {
  fftMiddleOut(out, in);
}

// out = in * (A - B)
void Gpu::tailMulDelta(TBuf& out, TBuf& in, TBuf& bufA, TBuf& bufB) {
  if (args.uses("NO_P2_FUSED_TAIL")) {
    fftHin(out, in);
    kernelMultiplyDelta(out, bufA, bufB);
    fftHout(out);
  } else {
    tailFusedMulDelta(out, in, bufA, bufB);
  }
}

vector<int> Gpu::readOut(ConstBuffer<int> &buf) {
  transposeOut(bufTranspose, buf);
  return bufTranspose.read();
}

void Gpu::writeIn(Buffer<int>& buf, const vector<u32>& words) { writeIn(buf, expandBits(words, N, E)); }

void Gpu::writeIn(Buffer<int>& buf, const vector<i32>& words) {
  bufTranspose.write(words);
  transposeIn(buf, bufTranspose);
}

namespace {

class IterationTimer {
  Timer timer;
  u32 kStart;

public:
  explicit IterationTimer(u32 kStart) : kStart(kStart) { }
  
  float reset(u32 k) {
    float secs = timer.deltaSecs();

    u32 its = max(1u, k - kStart);
    kStart = k;
    return secs / its;
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
    float secsPerIt = timer.reset(k);
    if (k % logStep == 0) { log("%u / %u, %.0f us/it\n", k, n, secsPerIt * 1'000'000); }
    if (k >= n) { break; }
  }
  return readData();
}

// A:= A^h * B
void Gpu::expMul(Buffer<i32>& A, u64 h, Buffer<i32>& B) {
  exponentiate(A, h, buf1, buf2, buf3);
  modMul(A, A, B, buf1, buf2, buf3);
}

// return A^x * B
Words Gpu::expMul(const Words& A, u64 h, const Words& B) {
  writeData(A);
  writeCheck(B);
  expMul(bufData, h, bufCheck);
  return readData();
}

Words Gpu::expMul2(const Words& A, u64 h, const Words& B) {
  expMul(A, h, B);
  modMul(bufData, bufData, bufCheck, buf1, buf2, buf3);
  return readData();
}

void Gpu::exponentiate(Buffer<int>& bufInOut, u64 exp, TBuf& buf1, TBuf& buf2, TBuf& buf3) {
  if (exp == 0) {
    bufInOut.set(1);
  } else if (exp > 1) {
    fftP(buf2, bufInOut);
    tW(buf3, buf2);
    fftHin(buf1, buf3);
    exponentiateCore(buf2, buf1, exp, buf3);
    fftW(buf3, buf2);
    carryA(bufInOut, buf3);
    carryB(bufInOut);
  }
}

// All buffers are in "low" position.
void Gpu::exponentiate(TBuf& out, const TBuf& base, u64 exp, TBuf& tmp1) {
  assert(exp > 1);
  exponentiateCore(out, base, exp, tmp1);
  doCarry(tmp1, out);
  tW(out, tmp1);
}

// All buffers are in "low" position.
void Gpu::exponentiateLow(TBuf& out, const TBuf& base, u64 exp, TBuf& tmp1, TBuf& tmp2) {
  assert(exp > 0);
  if (exp == 1) {
    out << base;
  } else {
    exponentiate(tmp2, base, exp, tmp1);
    fftHin(out, tmp2);
  }
}

namespace {
bool testBit(u64 x, int bit) { return x & (u64(1) << bit); }
}

// See "left-to-right binary exponentiation" on wikipedia
void Gpu::exponentiateCore(TBuf& out, const TBuf& base, u64 exp, TBuf& tmp) {
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

// does either carrryFused() or the expanded version depending on useLongCarry
void Gpu::doCarry(TBuf& out, TBuf& in) {
  if (useLongCarry) {
    fftW(out, in);
    carryA(in, out);
    carryB(in);
    fftP(out, in);
  } else {
    carryFused(out, in);
  }
}

void Gpu::coreStep(Buffer<int>& out, Buffer<int>& in, bool leadIn, bool leadOut, bool mul3) {
  if (leadIn) {
    fftP(buf2, in);
    tW(buf1, buf2);    
  }
  
  tailSquare(buf2, buf1);
  
  tH(buf1, buf2);

  if (leadOut) {
    fftW(buf2, buf1);
    if (mul3) { carryM(out, buf2); } else { carryA(out, buf2); }
    carryB(out);
  } else {
    assert(!useLongCarry);
    if (mul3) { carryFusedMul(buf2, buf1); } else { carryFused(buf2, buf1); }
    tW(buf1, buf2);
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

Stats Gpu::readStats(Buffer<int>& buf) {
  bufStatsOut.zero();
  stats(bufStatsOut, buf);
  auto s = bufStatsOut.read();
  assert(s.size() == 3);
  u32 sumM31 = modM31(s[2]);
  return {u64(s[0]), s[1], sumM31};
}

bool Gpu::equalNotZero(Buffer<int>& buf1, Buffer<int>& buf2) {
  Stats statsA = readStats(buf1);
  Stats statsB = readStats(buf2);
  return statsA.sumAbs && statsB.sumAbs && statsA.sumM31 == statsB.sumM31;
}

u64 Gpu::bufResidue(Buffer<int> &buf) {
  u32 earlyStart = N/2 - 32;
  vector<int> readBuf = readSmall(buf, earlyStart);
  return residueFromRaw(N, E, readBuf);
}

static string formatETA(u32 secs) {
  u32 etaMins = (secs + 30) / 60;
  int days  = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins  = etaMins % 60;
  char buf[64];
  if (days) {
    snprintf(buf, sizeof(buf), "%dd %02d:%02d", days, hours, mins);
  } else {
    snprintf(buf, sizeof(buf), "%02d:%02d", hours, mins);
  }
  return string(buf);  
}

static string getETA(u32 step, u32 total, float secsPerStep) {
  u32 etaSecs = max(0u, u32((total - step) * secsPerStep));
  return formatETA(etaSecs);
}

static string makeLogStr(const string& status, u32 k, u64 res, float secsPerIt, float secsCheck, float secsSave, u32 nIters) {
  char buf[256];
  
  snprintf(buf, sizeof(buf), "%2s %9u %6.2f%% %s %4.0f us/it + check %.2fs + save %.2fs; ETA %s",
           status.c_str(), k, k / float(nIters) * 100, hex(res).c_str(),
           secsPerIt * 1'000'000, secsCheck, secsSave, getETA(k, nIters, secsPerIt).c_str());
  return buf;
}

static void doBigLog(u32 E, u32 k, u64 res, bool checkOK, float secsPerIt, float secsCheck, float secsSave, u32 nIters, u32 nErrors, u32 nBitsP1, u32 B1, u64 resP1) {
  char buf[64] = {0};
  if (k < nBitsP1) {
    snprintf(buf, sizeof(buf), " | P1(%s) %2.1f%% ETA %s %016" PRIx64,
             formatBound(B1).c_str(), float(k) * 100 / nBitsP1, getETA(k, nBitsP1, secsPerIt).c_str(), resP1);
  }
  
  log("%s%s%s\n", makeLogStr(checkOK ? "OK" : "EE", k, res, secsPerIt, secsCheck, secsSave, nIters).c_str(),
      (nErrors ? " "s + to_string(nErrors) + " errors"s : ""s).c_str(), buf);
}

bool Gpu::equals9(const Words& a) {
  if (a[0] != 9) { return false; }
  for (auto it = next(a.begin()); it != a.end(); ++it) { if (*it) { return false; }}
  return true;
}

static u32 mod3(const std::vector<u32> &words) {
  u32 r = 0;
  // uses the fact that 2**32 % 3 == 1.
  for (u32 w : words) { r += w % 3; }
  return r % 3;
}

static void doDiv3(u32 E, Words& words) {
  u32 r = (3 - mod3(words)) % 3;
  assert(r < 3);
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

void Gpu::printRoundoff() {
  readROE();
  vector<u32> roeData = bufRoundoff.read();
  u32 nIts = roeData.back();
  roeData.pop_back();

  vector<float> roe;
  for (u32 x : roeData) {
    if (x) { roe.push_back(as<float>(x)); }
  }

  u32 N = roe.size();
  if (!N) { return; }

  sort(roe.begin(), roe.end());
    
#if 0
  {
    File fo = File::openAppend("roundoff.txt");
    if (fo) {
      for (float x : roe) { fprintf(fo.get(), "%f\n", x); }
      fprintf(fo.get(), "\n\n");
    }
  }
#endif

  /*
  double sum = 0;
  for (float x : roe) { sum += x; }
  double mean = sum / N;
  */

  // work out the mode
#if 0
  vector<u32> buckets(2000);
  u32 best = 0;
  for (float x : roe) {
    assert(0 <= x && x <= 0.5);
    int b = min(int(buckets.size()) - 1, int(x * 2000));
    ++buckets[b];
    if (buckets[b] > buckets[best]) { best = b; }
  }
  float mode = (best + 0.5f) / 2000;
#endif

  float mode = 0;
  u32 nMode = 0;
  float current = 0;
  u32 nCurrent = 0;
  u32 nBelow = 0;
  float m = 0;
  for (float x : roe) {
    if (x > current) {
      m = max(m, x);
      if (nCurrent >= nMode) {
        nBelow += nMode;
        mode = current;
        nMode = nCurrent;
      }
      current = x;
      nCurrent = 0;
    }
    ++nCurrent;
  }
  u32 nAbove = N - (nMode + nBelow);

  /*
  float median = N%2 ? roe[(N-1) / 2] : (roe[N/2-1] + roe[N/2]) / 2;
  double rightVariance = 0;
  u32 rightN = 0;
  u32 nMedian = 0;
  for (float x : roe) {
    if (x >= median) {
      if (x <= median) {
        ++nMedian;
      } else {
        ++rightN;
        m = max(m, x);
        double d = x - median;
        rightVariance += d * d;
      }
    }
  }
  rightVariance /= (rightN + nMedian / 2);
  double rightSdev = sqrt(rightVariance);
  double rightZ = (0.5f - median) / rightSdev;
  */

  log("ROE: N=%u (%u); mode %f (below %u, at %u, above %u), max %f, %f\n",
      N, nIts, mode, nBelow, nMode, nAbove, m, mode * (nMode + nAbove - nBelow) / nMode);

#if 0
  double gamma = 0.577215665; // Euler-Mascheroni
  double negloglog2 = -log(log(2)); // 0.366513;

  // estimate u, beta from mean and st.dev
  double beta1 = sqrt(6.0) / M_PI * sdev;
  double u1 = mean - beta1 * gamma;
  
  // estimate u, beta from mode and median
  double u2 = mode;
  double beta2 = (median - u2) / negloglog2;

  // double z = (0.5 - avg) / sdev;

  // See Gumbel distribution https://en.wikipedia.org/wiki/Gumbel_distribution
  // double p = -expm1(-exp(-z * (M_PI / sqrt(6))) * (E * exp(-gamma)));
  log("ROE: N=%u (%u), mode %f, median %f, mean %f, sdev %f, max %f, (%f, %f) (%f, %f) %u\n",
      N, nIts, mode, median, mean, sdev, m, u1, beta1, u2, beta2, buckets[best]);
#endif
}

void Gpu::accumulate(Buffer<int>& acc, TBuf& data, TBuf& tmp1, TBuf& tmp2) {
  fftP(tmp1, acc);
  tW(tmp2, tmp1);
  tailMul(tmp1, data, tmp2);
  tH(tmp2, tmp1);
  fftW(tmp1, tmp2);
  carryA(acc, tmp1);
  carryB(acc);
}

void Gpu::square(Buffer<int>& data, TBuf& tmp1, TBuf& tmp2) {
  fftP(tmp1, data);
  tW(tmp2, tmp1);
  tailSquare(tmp1, tmp2);
  tH(tmp2, tmp1);
  fftW(tmp1, tmp2);
  carryA(data, tmp1);
  carryB(data);  
}

void Gpu::square(Buffer<int>& data) {
  square(data, buf1, buf2);
}

// io *= in; with buffers in low position.
void Gpu::multiplyLowLow(TBuf& io, const TBuf& in, TBuf& tmp) {
  tailMulLowLow(io, in);
  tH(tmp, io);
  doCarry(io, tmp);
  tW(tmp, io);
  fftHin(io, tmp);
}

struct SquaringSet {  
  std::string name;
  u32 N;
  TBuf A, B, C;
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
  
  SquaringSet(Gpu& gpu, u32 N, const TBuf& bufBase, TBuf& bufTmp, TBuf& bufTmp2, array<u64, 3> exponents, string_view name)
    : SquaringSet(gpu, N, name) {
    
    gpu.exponentiateLow(C, bufBase, exponents[0], bufTmp, bufTmp2);
    gpu.exponentiateLow(A, bufBase, exponents[2], bufTmp, bufTmp2);    
    
    if (exponents[1] == exponents[2]) {
      B << A;
    } else {
      gpu.exponentiateLow(B, bufBase, exponents[1], bufTmp, bufTmp2);
    }
  }

  SquaringSet& operator=(const SquaringSet& rhs) {
    assert(N == rhs.N);
    copyFrom(rhs);
    return *this;
  }

  void step(TBuf& bufTmp) {
    gpu.multiplyLowLow(C, B, bufTmp);
    gpu.multiplyLowLow(B, A, bufTmp);
  }

private:
  void copyFrom(const SquaringSet& rhs) {
    A << rhs.A;
    B << rhs.B;
    C << rhs.C;
  }
};

template<typename Future> bool finished(const Future& f) {
  return f.valid() && f.wait_for(chrono::steady_clock::duration::zero()) == future_status::ready;
}

template<typename Future> bool wait(const Future& f) {
  if (f.valid()) {
    f.wait();
    return true;
  } else {
    return false;
  }
}

bool Gpu::verifyP2Checksums(const vector<TBuf>& bufs, const vector<u64>& sums) {
  // Timer timer;
  assert(bufs.size() == sums.size());
  bool ok = true;
  for (u32 i = 0, end = bufs.size(); i < end; ++i) {
    sum64(bufSumOut, N * 8, bufs[i]);
    u64 sum = bufSumOut.read()[0];
    if (sum != sums[i]) {
      log("EE checksum mismatch in P2 buf #%u: %" PRIx64 " vs. %" PRIx64 "\n", i, sum, sums[i]);
      ok = false;
    }
  }
  // log("%s buffer validation took %.1fs\n", ok ? "OK" : "EE", timer.deltaSecs());
  return ok;
}

bool Gpu::verifyP2Block(u32 D, const Words& p1Data, u32 block, const TBuf& bigC, Buffer<int>& bufP2Data) {
  Timer timer;
  tailSquareLow(buf1, bigC);
  tH(buf2, buf1);
  fftW(buf1, buf2);
  carryA(bufP2Data, buf1);
  carryB(bufP2Data);
  u64 resA = bufResidue(bufP2Data);

  writeIn(bufP2Data, p1Data);
  exponentiate(bufP2Data, u64(4 * D * D) * block * block, buf1, buf2, buf3);
  u64 resB = bufResidue(bufP2Data);

  bool ok = (resA == resB);
  if (ok) {
    log("OK @%u: %016" PRIx64 " (%.1fs)\n", block, resA, timer.deltaSecs());
  } else {
    log("EE @%u: %016" PRIx64 " vs. %016" PRIx64 " (%.1fs)\n", block, resA, resB, timer.deltaSecs());
  }
  return ok;
}

void Gpu::doP2(Saver* saver, u32 b1, u32 b2, future<string>& gcdFuture, Signal &signal) {
  if (!b1) { return; }
  assert(b2 && b2 > b1);
  
  u32 bufSize = N * sizeof(double);
  u32 nBuf = AllocTrac::availableBytes() / bufSize - 5;
  u32 D = Pm1Plan::getD(args.D, nBuf);
  LogContext pushContext{"P2("s + formatBound(b1) + ',' + formatBound(b2) + ")"};

  if (saver->loadP2(b2, D, nBuf) == u32(-1)) {
    // log("already finished\n");
    return;
  }

  log("D=%u, nBuf=%u\n", D, nBuf);
    
  Pm1Plan plan{args.D, nBuf, b1, b2};
  
  log("Generating P2 plan, please wait..\n");
  auto [beginBlock, selected] = plan.makePlan();
  
  bool printStats = args.flags.count("STATS");

 retry:
  u32 startBlock = saver->loadP2(b2, D, nBuf);
  if (!startBlock) { startBlock = beginBlock; }
  assert(beginBlock <= startBlock && startBlock < selected.size());
  log("%u blocks: %u - %u; start from %u\n", u32(selected.size()) - beginBlock, beginBlock, u32(selected.size()) - 1, startBlock);
  
  Memlock memlock{args.masterDir, u32(args.device)};
  Timer timer;

  vector<TBuf> blockBufs;
  const vector<u32>& jset = plan.jset;
  assert(jset.size() >= 24 && jset[0] == 1);
  
  for (u32 j : jset) { blockBufs.emplace_back(queue, "p2-"s + std::to_string(j), N); }
  log("Allocated %u buffers\n", u32(jset.size()));

  TBuf bufAcc{queue, "Acc", N};  // Second-stage accumulator.
  Buffer<int> bufP2Data{queue, "p2Data", N};

  u64 res64LittleA = 0;

  Words p1Data = saver->loadP1Final();
  
  {
    writeIn(bufP2Data, p1Data);
    exponentiate(bufP2Data, u64(4) * jset.back() * jset.back(), buf1, buf2, buf3);
    res64LittleA = bufResidue(bufP2Data);

    writeIn(bufP2Data, p1Data);
    fftP(buf2, bufP2Data);
    tW(buf1, buf2);
    tailSquare(buf2, buf1);
    tH(bufAcc, buf2);  // Save bufAcc for later use as accumulator

    // buf3 takes "data" in low position, used in initializing the SquaringSets below.
    doCarry(buf3, bufAcc);
    tW(buf1, buf3);
    fftHin(buf3, buf1);
  
    assert(!gcdFuture.valid());
    log("Starting P1 GCD\n");
    gcdFuture = async(launch::async, [E=E, p1Data]() { return GCD(E, p1Data, 1); });
  }

  vector<u64> blockChecksum(blockBufs.size());
  
  {
    u32 beginJ = jset[0];
    assert(beginJ == 1);    
    SquaringSet little{*this, N, buf3, buf1, buf2, {beginJ*beginJ, 4 * (beginJ + 1), 8}, "little"};
    
    for (u32 i = 0; i < jset.size(); ++i) {
      int delta = i ? jset[i] - jset[i-1] : 0;
      assert(delta % 2 == 0);
      for (int s = delta / 2; s > 0; --s) { little.step(buf1); }
      blockBufs[i] << little.C;
      sum64(bufSumOut, N * 8, blockBufs[i]);
      blockChecksum[i] = bufSumOut.read()[0];
    }

    tailSquareLow(buf1, little.C);
    tH(buf2, buf1);
    fftW(buf1, buf2);
    carryA(bufP2Data, buf1);
    carryB(bufP2Data);
    u64 res64LittleB = bufResidue(bufP2Data);
    if (res64LittleA != res64LittleB) {
      log("EE mismatch after little steps: %s vs. %s\n", hex(res64LittleB).c_str(), hex(res64LittleA).c_str());
      goto retry;
    }
  }

  // Let's do it once more to validate the checksums.
  {
    u32 beginJ = jset[0];
    SquaringSet little{*this, N, buf3, buf1, buf2, {beginJ*beginJ, 4 * (beginJ + 1), 8}, "little"};
    
    for (u32 i = 0; i < jset.size(); ++i) {
      int delta = i ? jset[i] - jset[i-1] : 0;
      assert(delta % 2 == 0);
      for (int s = delta / 2; s > 0; --s) { little.step(buf1); }
      blockBufs[i] << little.C;
    }
    if (!verifyP2Checksums(blockBufs, blockChecksum)) { goto retry; }
  }
  

  // Warn: hack: the use of buf1 below as both output and temporary relies on the implementation of exponentiateLow().
  exponentiateLow(buf1, buf3, plan.D * plan.D, buf1, buf2); // base^(D^2)  
  SquaringSet big{*this, N, buf1, buf2, buf3, {u64(startBlock)*startBlock, 2 * startBlock + 1, 2}, "big"};
  
  queue->finish();
  log("Setup %u P2 buffers in %.1fs\n", u32(jset.size()), timer.deltaSecs());

  bool ok = verifyP2Block(plan.D, p1Data, startBlock, big.C, bufP2Data);
  if (!ok) {
    log("Initial block verification failed\n");
    throw "EE P2 initial verification";
  }
  
  // ----

  u32 doneMuls = (startBlock - beginBlock) * 2;
  for (u32 b = beginBlock; b < startBlock; ++b) { doneMuls += selected[b].count(); }
  
  u32 leftMuls = (selected.size() - startBlock) * 2;
  for (u32 b = startBlock; b < selected.size(); ++b) { leftMuls += selected[b].count(); }

  log("MULs: done %u, left %u; %.1f%%\n", doneMuls, leftMuls, doneMuls * 100.0f / (doneMuls + leftMuls));
  
  timer.reset();

  u32 nMuls = 0;
  const u32 blockMulti = 20;
  Timer sinceLastGCD;
  float lastGCDduration = 0;

  for (u32 block = startBlock; block < selected.size(); ++block) {
    const auto& bits = selected[block];

    for (u32 i = 0; i < jset.size(); ++i) {
      if (bits[i]) {
        doCarry(buf1, bufAcc);
        tW(bufAcc, buf1);
        tailMulDelta(buf1, bufAcc, big.C, blockBufs[i]);
        tH(bufAcc, buf1);
      }
    }

    nMuls += bits.count() + 2;
    big.step(buf1);

    if (block % blockMulti == 0) {
      if (!args.noSpin) { spin(); }      
      queue->finish();
    }
    
    u32 nStop = signal.stopRequested();    
    bool atEnd = block == selected.size() - 1;
    bool doLog = atEnd || nStop || block % (20 * blockMulti) == 0;

    if (doLog) {
      if (true || printStats) { printRoundoff(); }

#if 0
      {
        fftW(buf1, bufAcc);
        carryA(bufP2Data, buf1);
        carryB(bufP2Data);
        Words p2Data = readAndCompress(bufP2Data);
        log("B2 at %u res64 %016lx\n", block, residue(p2Data));
      }
#endif 

      if (nMuls >= 100) {
        doneMuls += nMuls;
        leftMuls -= nMuls;
        [[maybe_unused]] float percent = doneMuls * 100.0f / (doneMuls + leftMuls);
        float secs = timer.deltaSecs();
        u32 etaSecs = secs * leftMuls / nMuls;
        log("%5.2f%% %4.0f ETA %s\n", percent, secs / nMuls * 1e6, formatETA(etaSecs).c_str());
        nMuls = 0;
      }
    }

    if ((nStop || atEnd) && gcdFuture.valid()) {
      log("waiting for GCD..\n");
      wait(gcdFuture);
    }
    
    if (finished(gcdFuture)) {
      lastGCDduration = sinceLastGCD.deltaSecs();
      string factor = gcdFuture.get();
      log("GCD : %s\n", factor.empty() ? "no factor" : factor.c_str());
      
      if (!factor.empty()) {
        assert(!gcdFuture.valid());
        gcdFuture = async([factor](){ return factor; });
        return;
      }

      if (nStop) {
        queue->finish();
        throw "stop requested";
      }
    }

    if (nStop > 1) {
      assert(!gcdFuture.valid());
      queue->finish();
      throw "stop requested";
    }
    
    bool doGCD = nStop || atEnd || (!gcdFuture.valid() && sinceLastGCD.elapsedSecs() > max(600.0f, 10*lastGCDduration));
    
    if (doGCD) {
      if (!verifyP2Checksums(blockBufs, blockChecksum)) {
        goto retry;
      }
      if (!verifyP2Block(plan.D, p1Data, block + 1, big.C, bufP2Data)) {
        goto retry;
      }
      log("Starting GCD\n");
      fftW(buf1, bufAcc);
      carryA(bufP2Data, buf1);
      carryB(bufP2Data);
      Words p2Data = readAndCompress(bufP2Data);
      if (p2Data.empty()) {
        log("P2 error: ZERO, will retry\n");
        goto retry;
      }
      assert(!gcdFuture.valid());
      const u32 nextBlock = atEnd ? u32(-1) : (block + 1);
      gcdFuture = async(launch::async, [E=E, b2, D, nBuf, nextBlock, p2Data=std::move(p2Data), saver]() {
        string factor = GCD(E, p2Data, 0);
        saver->saveP2(b2, D, nBuf, nextBlock);
        return factor;
      });
      sinceLastGCD.reset();
    }
  }
  queue->finish();
}

// ----

namespace {

struct JacobiResult {
  bool ok;
  u32 k;
  u64 res64;
};

JacobiResult doJacobiCheck(u32 E, const Words& data, u32 k) {
  return {jacobi(E, data) == 1, k, res64(data)};
}

}

fs::path Gpu::saveProof(const Args& args, const ProofSet& proofSet) {
  Memlock memlock{args.masterDir, u32(args.device)};
  
  for (int retry = 0; retry < 2; ++retry) {
    Proof proof = proofSet.computeProof(this);
    fs::path tmpFile = proof.file(args.proofToVerifyDir);
    proof.save(tmpFile);
            
    fs::path proofFile = proof.file(args.proofResultDir);            
    bool doVerify = proofSet.power >= args.proofVerify;
    bool ok = !doVerify || Proof::load(tmpFile).verify(this);
    if (doVerify) { log("Proof '%s' verification %s\n", tmpFile.string().c_str(), ok ? "OK" : "FAILED"); }
    if (ok) {
      error_code noThrow;
      fs::remove(proofFile, noThrow);
      fs::rename(tmpFile, proofFile);
      log("Proof '%s' generated\n", proofFile.string().c_str());
      return proofFile;
    }
  }
  throw "bad proof generation";
}

PRPResult Gpu::isPrimePRP(const Args &args, const Task& task) {
  u32 E = task.exponent;
  u32 b1 = task.B1;
  u32 b2 = task.B2;
  u32 k = 0, blockSize = 0;
  u32 nErrors = 0;

  // log("maxAlloc: %.1f GB\n", args.maxAlloc * (1.0f / (1 << 30)));
  if (!args.maxAlloc) {
    // log("You should use -maxAlloc if your GPU has more than 4GB memory. See help '-h'\n");
  }
  
  u32 power = -1;
  u32 startK = 0;

  Saver saver{E, args.nSavefiles, b1, args.startFrom};
  B1Accumulator b1Acc{this, &saver, E};
  future<string> gcdFuture;
  future<JacobiResult> jacobiFuture;
  Signal signal;

  // Used to detect a repetitive failure, which is more likely to indicate a software rather than a HW problem.
  std::optional<u64> lastFailedRes64;

  // Number of sequential errors (with no success in between). If this ever gets high enough, stop.
  int nSeqErrors = 0;
  
 reload:
  {
    PRPState loaded = saver.loadPRP(args.blockSize);
    b1Acc.load(loaded.k);
    
    writeState(loaded.check, loaded.blockSize, buf1, buf2, buf3);
    
    u64 res = dataResidue();
    auto data = readData();

    if (res == loaded.res64) {
      log("OK %9u on-load: blockSize %d, %016" PRIx64 "\n", loaded.k, loaded.blockSize, res);
      // On the OK branch do not clear lastFailedRes64 -- we still want to compare it with the GEC check.
    } else {
      log("EE %9u on-load: %016" PRIx64 " vs. %016" PRIx64 "\n", loaded.k, res, loaded.res64);
      if (lastFailedRes64 && res == *lastFailedRes64) {
        throw "error on load";
      }
      lastFailedRes64 = res;
      goto reload;
    }
    
    k = loaded.k;
    blockSize = loaded.blockSize;
    if (nErrors == 0) { nErrors = loaded.nErrors; }
    assert(nErrors >= loaded.nErrors);
  }

  if (k) {
    Words b1Data = b1Acc.fold();
    if (!b1Data.empty()) {
      // log("P1 %9u starting on-load Jacobi check\n", k);
      jacobiFuture = async(launch::async, doJacobiCheck, E, std::move(b1Data), k);
    }
  }

  assert(blockSize > 0 && 10000 % blockSize == 0);
  
  u32 checkStep = checkStepForErrors(args.logStep, nErrors);
  assert(checkStep % 10000 == 0);

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

  bool isPrime = false;
  IterationTimer iterationTimer{startK};

  u64 finalRes64 = 0;

  // We extract the res64 at kEnd.
  // For M=2^E-1, residue "type-3" == 3^(M+1), and residue "type-1" == type-3 / 9,
  // See http://www.mersenneforum.org/showpost.php?p=468378&postcount=209
  // For both type-1 and type-3 we need to do E squarings (as M+1==2^E).
  const u32 kEnd = E;
  assert(k < kEnd);

  // We continue beyound kEnd: up to the next multiple of 1024 if proof is enabled (kProofEnd), and up to the next blockSize
  u32 kEndEnd = roundUp(kEnd, blockSize);

  bool printStats = args.flags.count("STATS");

  bool skipNextCheckUpdate = false;

  u32 persistK = proofSet.next(k);
  bool leadIn = true;

  assert(k % blockSize == 0);
  assert(checkStep % blockSize == 0);

  bool didP2 = false;
  
  while (true) {
    assert(k < kEndEnd);

    if (finished(jacobiFuture)) {
      auto [ok, jacobiK, res] = jacobiFuture.get();
      log("P1 Jacobi %s @ %u %016" PRIx64 "\n", ok ? "OK" : "EE", jacobiK, res);      
      if (!ok) {
        if (jacobiK < k) {
          saver.deleteBadSavefiles(jacobiK, k);
          ++nErrors;
          goto reload;
        }
      }
    }
    
    if (skipNextCheckUpdate) {
      skipNextCheckUpdate = false;
    } else if (k % blockSize == 0) {
      if (leadIn) {
        modMul(bufCheck, bufCheck, bufData, buf1, buf2, buf3);
      } else {
        mul(bufCheck, buf1);
      }
    }

    ++k; // !! early inc
    assert(b1Acc.wantK() == 0 || b1Acc.wantK() >= k);

    bool doStop = false;
    bool b1JustFinished = false;

    if (k % blockSize == 0) {
      queue->finish(); // Avoid loading too much work ahead of time on the queue.
      doStop = signal.stopRequested() || (args.iters && k - startK >= args.iters);
      b1JustFinished = !b1Acc.wantK() && !didP2 && !jacobiFuture.valid() && (k - startK >= 2 * blockSize);
    }
    
    bool leadOut = doStop || b1JustFinished || (k % 10000 == 0) || (k % blockSize == 0 && k >= kEndEnd) || k == persistK || k == kEnd || useLongCarry;

    coreStep(bufData, bufData, leadIn, leadOut, false);
    leadIn = leadOut;

#if 0
    log("k %d %016" PRIx64 " ", k, dataResidue());
    printRoundoff();

    vector<int> tmp = readOut(bufData);
    auto [sumAbs, sum] = sumStats(tmp);
    u32 m1 = modM31(N, E, tmp);
    log("cpu: %08x %lu %ld \n", m1, sumAbs, sum);
    bufStatsOut.zero();
    stats(bufStatsOut, bufData);
    auto s = bufStatsOut.read();
    assert(s.size() == 3);
    u32 m2 = modM31(u64(s[2]));
    log("gpu: %08x %ld %ld\n", m2, s[0], s[1]);

    auto trip = expandBits(compactBits(tmp, E), N, E);
    tie(sumAbs, sum) = sumStats(trip);
    u32 m3 = modM31(N, E, trip);
    log("cpu: %08x %lu %ld\n", m3, sumAbs, sum);

    assert(m1 == m2 && m2 == m3);

    writeIn(bufData, trip);
#endif
    
    if (k == persistK) {
      Words data = readData();
      if (data.empty()) {
        log("Data error ZERO\n");
        ++nErrors;
        goto reload;
      }
      proofSet.save(k, data);
      persistK = proofSet.next(k);
    }

    if (k == kEnd) {
      auto words = readData();
      isPrime = equals9(words);
      doDiv9(E, words);
      finalRes64 = residue(words);
      log("%s %8d / %d, %s\n", isPrime ? "PP" : "CC", kEnd, E, hex(finalRes64).c_str());
    }

    if (k == b1Acc.wantK()) {
      if (leadOut) {
        b1Acc.step(k, bufData);
      } else {
        b1Acc.step(k, buf1);
      }
      assert(!b1Acc.wantK() || b1Acc.wantK() > k);
    }

    if (!leadOut) {
      if (k % blockSize == 0) {
        finish();
        if (!args.noSpin) { spin(); }
      }
      continue;
    }

    bool doCheck = doStop || b1JustFinished || (k % checkStep == 0) || (k >= kEndEnd) || (k - startK == 2 * blockSize);

    if (k % 10000 == 0) {
      Stats stats = readStats(bufData);
      float secsPerIt = iterationTimer.reset(k);
      float avgBits = log2f(stats.sumAbs / N) + 1;
      log("   %9u %08x  bits %.2f, balance %d; %4.0f\n", k, stats.sumM31, avgBits, int(stats.sum), secsPerIt * 1'000'000);
      printRoundoff();
      if (stats.sumAbs == 0) { doCheck = true; }
    }

    /*
    if (k % 10000 == 0 && !doCheck) {
      float secsPerIt = iterationTimer.reset(k);
      // log("   %9u %6.2f%% %s %4.0f us/it\n", k, k / float(kEndEnd) * 100, hex(res).c_str(), secsPerIt * 1'000'000);
      log("%9u %s %4.0f\n", k, hex(res).c_str(), secsPerIt * 1'000'000);
    }
    */

    if (doStop) {
      log("Stopping, please wait..\n");
      signal.release();
      wait(gcdFuture);
    }
      
    if (finished(gcdFuture)) {
      string factor = gcdFuture.get();
      log("GCD: %s\n", factor.empty() ? "no factor" : factor.c_str());
      
      assert(didP2);
      if (didP2) { task.writeResultPM1(args, factor, getFFTSize()); }
      
      if (!factor.empty()) { return {factor}; }
    }
      
    if (doCheck) {
      u64 res = dataResidue();
      float secsPerIt = iterationTimer.reset(k);

      u32 checkSum = 0;
      Words check = readAndCompress(bufCheck, &checkSum);
      // log("check %08x\n", checkSum);
      if (check.empty()) { log("Check read ZERO\n"); }

      // Round-trip
      vector<i32> expanded = expandBits(check, N, E);
      assert(checkSum == modM31(N, E, expanded));
      writeIn(bufCheck, expanded);

      bool ok = !check.empty() && this->doCheck(blockSize, buf1, buf2, buf3);

      float secsCheck = iterationTimer.reset(k);
        
      if (ok) {
        nSeqErrors = 0;
        lastFailedRes64.reset();
        skipNextCheckUpdate = true;

        Words b1Data;
        try {
          b1Data = b1Acc.save(k);
        } catch (Reload&) {
          goto reload;
        }

        if (k < kEnd) { saver.savePRP(PRPState{k, blockSize, res, check, nErrors}); }

        float secsSave = iterationTimer.reset(k);
          
        doBigLog(E, k, res, ok, secsPerIt, secsCheck, secsSave, kEndEnd, nErrors, b1Acc.nBits, b1Acc.b1, ::res64(b1Data));

        if (!b1Data.empty() && (!b1Acc.wantK() || (k % 1'000'000 == 0)) && !jacobiFuture.valid()) {
          // log("P1 %9u starting Jacobi check\n", k);
          jacobiFuture = async(launch::async, doJacobiCheck, E, std::move(b1Data), k);
        }

        if (!doStop && !didP2 && !b1Acc.wantK() && !jacobiFuture.valid()) {
          doP2(&saver, b1, b2, gcdFuture, signal);
          didP2 = true;
        }
          
        if (k >= kEndEnd) {
          fs::path proofFile = saveProof(args, proofSet);
          return {"", isPrime, finalRes64, nErrors, proofFile.string()};          
        }
        
      } else {
        doBigLog(E, k, res, ok, secsPerIt, secsCheck, 0, kEndEnd, nErrors, b1Acc.nBits, b1Acc.b1, 0);
        ++nErrors;
        if (++nSeqErrors > 2) {
          log("%d sequential errors, will stop.\n", nSeqErrors);
          throw "too many errors";
        }
        if (lastFailedRes64 && (res == *lastFailedRes64)) {
          log("Consistent error %016" PRIx64 ", will stop.\n", res);
          throw "consistent error";
        }
        lastFailedRes64 = res;
        if (!doStop) { goto reload; }
      }
        
      logTimeKernels();
        
      if (doStop) {
        assert(!gcdFuture.valid());
        queue->finish();
        throw "stop requested";
      }
        
      iterationTimer.reset(k);
    }
  }
}

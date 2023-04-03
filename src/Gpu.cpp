// Copyright Mihai Preda and George Woltman.

#include "Gpu.h"
#include "Proof.h"
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
#include <array>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static_assert(sizeof(double2) == 16, "size double2");
static_assert(sizeof(long double) > sizeof(double), "long double offers extended precision");

extern const char *CL_SOURCE;

using float3 = tuple<float, float, float>;

struct Weights {
  vector<double> threadWeightsIF;  
  vector<double> carryWeightsIF;
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

double2 *smallTrigBlock(u32 W, u32 H, double2 *p) {
  for (u32 line = 1; line < H; ++line) {
    for (u32 col = 0; col < W; ++col) {
      *p++ = root1<double>(W * H, line * col);
    }
  }
  return p;
}

ConstBuffer<double2> genSmallTrig(const Context& context, u32 size, u32 radix) {
  vector<double2> tab;

  // smallTrigBlock(size / radix, 2, tab.data());

  for (u32 line = 1; line < radix; ++line) {
  for (u32 col = 0; col < size / radix; ++col) {
    tab.push_back(root1<double>(size, col * line));
  }
  }
  tab.resize(size);
  
  /*
  auto *p = tab.data() + radix;
  for (u32 w = radix; w < size; w *= radix) { p = smallTrigBlock(w, std::min(radix, size / w), p); }
  assert(p - tab.data() == size);
  */
  return {context, "smallTrig", tab};
}

ConstBuffer<double2> genMiddleTrig(const Context& context, u32 smallH, u32 middle) {
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
    auto[c, s] = root1<f128>(hN, k);
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

double boundUnderOne(double x) { return std::min(x, nexttoward(1, 0)); }

#define CARRY_LEN 8

Weights genWeights(u32 E, u32 W, u32 H, u32 nW) {
  u32 N = 2u * W * H;
  
  u32 groupWidth = W / nW;

  // Inverse + Forward
  vector<double> threadWeightsIF;
  for (u32 thread = 0; thread < groupWidth; ++thread) {
    auto iw = invWeight(N, E, H, 0, thread, 0);
    threadWeightsIF.push_back(iw - 1);
    auto w = weight(N, E, H, 0, thread, 0);
    threadWeightsIF.push_back(w - 1);
  }

  // Inverse only. Also the group order matches CarryA/M (not fftP/CarryFused).
  vector<double> carryWeightsIF;
  for (u32 gy = 0; gy < H / CARRY_LEN; ++gy) {
    auto iw = invWeight(N, E, H, gy * CARRY_LEN, 0, 0);
    carryWeightsIF.push_back(2 * boundUnderOne(iw));
    
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
    for (u32 gx = 0; gx < nW; ++gx) {
      for (u32 thread = 0; thread < groupWidth; ) {
        std::bitset<32> b;
        for (u32 bitoffset = 0; bitoffset < 32; bitoffset += CARRY_LEN * 2, ++thread) {
          for (u32 block = 0; block < CARRY_LEN; ++block) {
            for (u32 rep = 0; rep < 2; ++rep) {
              if (isBigWord(N, E, kAt(H, gy * CARRY_LEN + block, gx * groupWidth + thread) + rep)) { b.set(bitoffset + block * 2 + rep); }
            }
          }
        }
        bitsC.push_back(b.to_ulong());
      }
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

[[maybe_unused]] string toLiteral(float3 v) {
  auto [a, b, c] = v;
  return "("s + toLiteral(a) + ',' + toLiteral(b) + ',' + toLiteral(c) + ')';
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

  // Force carry64 when carry32 might exceed a very conservative 0x6C000000
  if (FFTConfig::getMaxCarry32(N, E) > 0x6C00) { defines.push_back({"CARRY64", 1}); }

  // If we are near the maximum exponent for this FFT, then we may need to set some chain #defines
  // to reduce the round off errors.
  auto [mm_chain, mm2_chain, ultra_trig] = FFTConfig::getChainLengths(N, E, MIDDLE);
  if (mm_chain) { defines.push_back({"MM_CHAIN", mm_chain}); }
  if (mm2_chain) { defines.push_back({"MM2_CHAIN", mm2_chain}); }
  if (ultra_trig) { defines.push_back({"ULTRA_TRIG", 1}); }

  defines.push_back({"WEIGHT_STEP", double(weight(N, E, SMALL_HEIGHT * MIDDLE, 0, 0, 1) - 1)});
  defines.push_back({"IWEIGHT_STEP", double(invWeight(N, E, SMALL_HEIGHT * MIDDLE, 0, 0, 1) - 1)});

  vector<double> iWeights;
  vector<double> fWeights;
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
  : Gpu{args, E, W, BIG_H, SMALL_H, nW, nH, device, timeKernels, useLongCarry, genWeights(E, W, BIG_H, nW)}
{}

using float2 = pair<float, float>;

#define ROE_SIZE 111000

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
  
  LOAD(kernCarryFused,    BIG_H + 1),
  LOAD(kernCarryFusedMul, BIG_H + 1),
  LOAD(fftP, BIG_H),
  LOAD(fftW,   BIG_H),
  LOAD(fftHin,  hN / SMALL_H),
  LOAD(fftHout, hN / SMALL_H),
  LOAD_WS(fftMiddleIn,  hN / (BIG_H / SMALL_H)),
  LOAD_WS(fftMiddleOut, hN / (BIG_H / SMALL_H)),
  LOAD_WS(kernCarryA,  hN / CARRY_LEN),
  LOAD_WS(kernCarryM,  hN / CARRY_LEN),
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
  LOAD(isNotZero, 256),
  LOAD(isEqual, 256),
  LOAD(sum64, 256),
#undef LOAD_WS
#undef LOAD

  bufTrigW{genSmallTrig(context, W, nW)},
  bufTrigH{genSmallTrig(context, SMALL_H, nH)},
  bufTrigM{genMiddleTrig(context, SMALL_H, BIG_H / SMALL_H)},
  bufBits{context, "bits", weights.bitsCF},
  bufBitsC{context, "bitsC", weights.bitsC},
  bufData{queue, "data", N},
  bufAux{queue, "aux", N},
  bufCheck{queue, "check", N},
  bufBase{queue, "base", N},
  bufCarry{queue, "carry", N / 2},
  bufReady{queue, "ready", BIG_H},
  bufCarryMax{queue, "carryMax", 8},
  bufCarryMulMax{queue, "carryMulMax", 8},
  bufSmallOut{queue, "smallOut", 256},
  bufSumOut{queue, "sumOut", 1},
  bufROE{queue, "ROE", ROE_SIZE},
  roePos{0},
  buf1{queue, "buf1", N},
  buf2{queue, "buf2", N},
  buf3{queue, "buf3", N},
  usesROE1{args.uses("ROE1")},
  usesROE2{args.uses("ROE2")},
  args{args}
{
  // dumpBinary(program.get(), "isa.bin");
  
  kernCarryFused.setFixedArgs(   3, bufCarry, bufReady, bufTrigW, bufBits, bufROE, bufCarryMax);
  kernCarryFusedMul.setFixedArgs(3, bufCarry, bufReady, bufTrigW, bufBits, bufROE, bufCarryMulMax);
  fftP.setFixedArgs(2, bufTrigW);
  fftW.setFixedArgs(2, bufTrigW);
  fftHin.setFixedArgs(2, bufTrigH);
  fftHout.setFixedArgs(1, bufTrigH);
  fftMiddleIn.setFixedArgs(2, bufTrigM);
  fftMiddleOut.setFixedArgs(2, bufTrigM);
    
  kernCarryA.setFixedArgs(3, bufCarry, bufBitsC, bufROE, bufCarryMax);
  kernCarryM.setFixedArgs(3, bufCarry, bufBitsC, bufROE, bufCarryMulMax);
  carryB.setFixedArgs(1, bufCarry, bufBitsC);

  tailFusedMulDelta.setFixedArgs(4, bufTrigH, bufTrigH);
  tailFusedMulLow.setFixedArgs(3, bufTrigH, bufTrigH);
  tailFusedMul.setFixedArgs(3, bufTrigH, bufTrigH);
  tailMulLowLow.setFixedArgs(2, bufTrigH);
  
  tailFusedSquare.setFixedArgs(2, bufTrigH, bufTrigH);
  tailSquareLow.setFixedArgs(2, bufTrigH, bufTrigH);

  bufReady.zero();
  bufCarryMax.zero();
  bufCarryMulMax.zero();

  vector<float2> readTrigSH, readTrigBH, readTrigN;
  {
    HostAccessBuffer<float2>
      bufSH{queue, "readTrig", SMALL_H/4 + 1},
      bufBH{queue, "readTrigBH", BIG_H/8 + 1},
      bufN{queue, "readTrigN", hN/8+1};
        
    Kernel{program.get(), queue, device, 32, "readHwTrig"}(bufSH, bufBH, bufN);
    readTrigSH = bufSH.read();
    readTrigBH = bufBH.read();
    readTrigN = bufN.read();

    Kernel{program.get(), queue, device, 32, "writeGlobals"}(ConstBuffer{context, "dp1", makeTrig<double>(2 * SMALL_H)},
                                                             ConstBuffer{context, "dp2", makeTrig<double>(BIG_H)},
                                                             ConstBuffer{context, "dp3", makeTrig<double>(hN)},
                                                             ConstBuffer{context, "dp4", makeTinyTrig<double>(W, hN)},

                                                             ConstBuffer{context, "w2", weights.threadWeightsIF},
                                                             ConstBuffer{context, "w3", weights.carryWeightsIF}
                                                             );
  }

  bufROE.zero();
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
      Buffer<double> A{queue, "A", N};
      Buffer<double> B{queue, "B", N};
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

ROEInfo norm(const vector<float>& v) {
  double acc = 0;
  float m = 0;
  for (float x : v) {
    assert(x >= 0 && x <= 0.5f);
    m = max(m, x);
    acc += x * x;
  }
  float n = v.empty() ? 0.0f : (sqrtf(float(acc) / u32(v.size())));
  return {u32(v.size()), m, n};
}

ROEInfo Gpu::readROE() {
  assert(roePos <= ROE_SIZE);
  if (roePos) {
    vector<float> roe = bufROE.read(roePos);
    assert(roe.size() == roePos);
    bufROE.zero(roePos);
    roePos = 0;
    return norm(roe);
  } else {
    return {};
  }
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

vector<u32> Gpu::readAndCompress(ConstBuffer<int>& buf)  {
  for (int nRetry = 0; nRetry < 3; ++nRetry) {
    sum64(bufSumOut, u32(buf.size * sizeof(int)), buf);
    
    vector<u64> expectedVect(1);
    bufSumOut.readAsync(expectedVect);
    vector<int> data = readOut(buf);
    u64 expectedSum = expectedVect[0];
    
    u64 sum = 0;
    bool allZero = true;
    for (auto it = data.begin(), end = data.end(); it < end; it += 2) {
      u64 v = u32(*it) | (u64(*(it + 1)) << 32);
      sum += v;
      allZero &= !v;
    }

    if (sum != expectedSum || (allZero && nRetry == 0)) {
      log("GPU -> Host read #%d failed (check %x vs %x)\n", nRetry, unsigned(sum), unsigned(expectedSum));
    } else {
      if (allZero) {
        log("Read ZERO\n");
        return {};
      } else {
        return compactBits(std::move(data),  E);
      }
    }
  }
  throw "Persistent read errors: GPU->Host";
}

vector<u32> Gpu::readCheck() { return readAndCompress(bufCheck); }
vector<u32> Gpu::readData() { return readAndCompress(bufData); }

void Gpu::tailMul(Buffer<double>& out, Buffer<double>& in, Buffer<double>& inTmp) {
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
void Gpu::mul(Buffer<int>& out, Buffer<int>& inA, Buffer<double>& inB, Buffer<double>& tmp1, Buffer<double>& tmp2, bool mul3) {
    fftP(tmp1, inA);
    tW(tmp2, tmp1);
    tailMul(tmp1, inB, tmp2);
    tH(tmp2, tmp1);
    fftW(tmp1, tmp2);
    if (mul3) { carryM(out, tmp1); } else { carryA(out, tmp1); }
    carryB(out);
}

// out := inA * inB;
void Gpu::modMul(Buffer<int>& out, Buffer<int>& inA, Buffer<int>& inB, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3, bool mul3) {

  fftP(buf1, inB);
  tW(buf3, buf1);
  
  mul(out, inA, buf3, buf1, buf2, mul3);
};

void Gpu::mul(Buffer<int>& io, Buffer<int>& inA, Buffer<int>& inB) { modMul(io, inA, inB, buf1, buf2, buf3); }

void Gpu::mul(Buffer<int>& io, Buffer<int>& inB) { mul(io, io, inB); }

void Gpu::mul(Buffer<int>& io, Buffer<double>& buf1) {
  // We know that coreStep() stores double output in buf1; so we're going to use buf2 & buf3 for temps.
  // tW(buf2, buf1);
  mul(io, io, buf1, buf2, buf3, false);
}

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

// out = in * (A - B)
void Gpu::tailMulDelta(Buffer<double>& out, Buffer<double>& in, Buffer<double>& bufA, Buffer<double>& bufB) {
  if (args.uses("NO_P2_FUSED_TAIL")) {
    fftHin(out, in);
    kernelMultiplyDelta(out, bufA, bufB);
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
  bufAux.write(words);
  transposeIn(buf, bufAux);
}

namespace {

class IterationTimer {
  Timer timer;
  u32 kStart;

public:
  explicit IterationTimer(u32 kStart) : kStart(kStart) { }
  
  float reset(u32 k) {
    float secs = timer.reset();

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

void Gpu::exponentiate(Buffer<int>& bufInOut, u64 exp, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
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
void Gpu::exponentiate(Buffer<double>& out, const Buffer<double>& base, u64 exp, Buffer<double>& tmp1) {
  assert(exp > 1);
  exponentiateCore(out, base, exp, tmp1);
  doCarry(tmp1, out);
  tW(out, tmp1);
}

// All buffers are in "low" position.
void Gpu::exponentiateLow(Buffer<double>& out, const Buffer<double>& base, u64 exp, Buffer<double>& tmp1, Buffer<double>& tmp2) {
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

static void doBigLog(u32 E, u32 k, u64 res, bool checkOK, float secsPerIt, float secsCheck, float secsSave, u32 nIters, u32 nErrors) {
  char buf[64] = {0};
  
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

#if 0
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
  bufRoundoff.write(zero);
  bufCarryMax.write(zero);
  bufCarryMulMax.write(zero);

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

  // #if 0
  u32 carryN = carry[3];
  u32 carryAvg = carryN ? *(u64*)&carry[0] / carryN : 0;
  u32 carryMax = carry[2];
  
  u32 carryMulN = carryMul[3];
  u32 carryMulAvg = carryMulN ? *(u64*)&carryMul[0] / carryMulN : 0;
  u32 carryMulMax = carryMul[2];

  log("Carry: N=%u, max %x, avg %x; CarryM: N=%u, max %x, avg %x\n",
      carryN, carryMax, carryAvg, carryMulN, carryMulMax, carryMulAvg);
  // #endif
}
#endif

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

void Gpu::square(Buffer<int>& data) {
  square(data, buf1, buf2);
}

// io *= in; with buffers in low position.
void Gpu::multiplyLowLow(Buffer<double>& io, const Buffer<double>& in, Buffer<double>& tmp) {
  tailMulLowLow(io, in);
  tH(tmp, io);
  doCarry(io, tmp);
  tW(tmp, io);
  fftHin(io, tmp);
}

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

  void step(Buffer<double>& bufTmp) {
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

// ----

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

/*
vector<bool> reverse(const vector<bool>& c) {
  vector<bool> ret;
  ret.reserve(c.size());
  for (auto it = c.end(), beg = c.begin(); it > beg;) { ret.push_back(*--it); }
  return ret;
}
*/

[[nodiscard]] vector<bool> addLE(const vector<bool>& a, const vector<bool>& b) {
  vector<bool> c;
  c.reserve(max(a.size(), b.size()) + 1);

  u32 carry = 0;
  auto ia = a.begin();
  auto ib = b.begin();
  for (; ia != a.end() && ib != b.end(); ++ia, ++ib) {
    u32 s = *ia + *ib + carry;
    c.push_back(s & 1u);
    carry = (s >> 1);
  }
  for (auto it = ia == a.end() ? ib : ia, end = ia == a.end() ? b.end() : a.end(); it != end; ++it) {
    u32 s = *it + carry;
    c.push_back(s & 1u);
    carry = (s >> 1);
  }
  if (carry) { c.push_back(1); }
  while (!c.empty() && !c.back()) { c.pop_back(); }
  return c;
}

vector<bool> takeTopBits(vector<bool>& v, u32 n) {
  assert(v.size() >= n);
  vector<bool> ret;
  ret.reserve(n);
  for (auto it = prev(v.end(), n), end = v.end(); it != end; ++it) {
    ret.push_back(*it);
  }
  v.resize(v.size() - n);
  return ret;
}

void Gpu::pm1Block(vector<bool> bitsLE, bool update) {
  if (update) {
    modMul(bufCheck, bufCheck, bufData, buf1, buf2, buf3);
  }

  bool leadIn = true;
  for (auto it = bitsLE.end(), beg = bitsLE.begin(); it > beg;) {
    --it;
    bool leadOut = (it == beg);
    coreStep(bufData, bufData, leadIn, leadOut, *it);
    leadIn = leadOut;
  }
}

bool Gpu::pm1Check(vector<bool> sumBits, u32 blockSize) {
  assert(!sumBits.empty() && sumBits.back());

  writeIn(bufAux, makeWords(E, 1));

  for (u32 i = sumBits.size(); i < blockSize; ++i) { sumBits.push_back(0); }
  assert(sumBits.size() >= blockSize);

  bool leadIn = true;
  for (int i = sumBits.size() - 1; i >= 0; --i) {
    // At this particular point we multiply-in bufCheck, which will thus suffer blockSize squarings in the end.
    if (u32(i) == blockSize - 1) {
      assert(leadIn); // we did a leadOut at the previous step if there was one.
      modMul(bufAux, bufAux, bufCheck, buf1, buf2, buf3);
    }

    bool leadOut = (i == 0) || (u32(i) == blockSize);
    coreStep(bufAux, bufAux, leadIn, leadOut, sumBits[i]);
    leadIn = leadOut;
  }

  modMul(bufAux, bufAux, bufBase, buf1, buf2, buf3);
  modMul(bufCheck, bufCheck, bufData, buf1, buf2, buf3);
  return equalNotZero(bufCheck, bufAux);
}

static void pm1Log(u32 B1, u32 k, u32 nBits, string strOK, u64 res64, float secsPerIt, float checkSecs, u32 nErr, ROEInfo roeInfo) {
  // char checkTimeStr[64] = {0};
  // if (checkSecs) { snprintf(checkTimeStr, sizeof(checkTimeStr), " (check %.0f ms)", checkSecs * 1000); }
  [[maybe_unused]] float percent = k * 100.0f / nBits;
  float us = secsPerIt * 1'000'000;
  // log("%7u/%u %5.2f%% %2s %016" PRIx64 " %4.0f\n",
  //    k, nBits, percent, strOK.c_str(), res64, us/*, checkTimeStr*/);
  // log("%7u %2s %016" PRIx64 " %4.0f\n", k, strOK.c_str(), res64, us);
  string err = nErr ? " err "s + to_string(nErr) : "";
  if (roeInfo.N) {
    log("%5.2f%% %1s %016" PRIx64 " %4.0f%s; ROE=%.3f %.4f %u\n",
        percent, strOK.c_str(), res64, us, err.c_str(),
        roeInfo.max, roeInfo.norm, roeInfo.N);
  } else {
    log("%5.2f%% %1s %016" PRIx64 " %4.0f%s\n",
        percent, strOK.c_str(), res64, us, err.c_str());
  }
}

bool Gpu::pm1Retry(const Args &args, const Task& task, u32 nErr) {
  enum RetCode { DONE=false, RETRY=true};
  const u32 blockSize = 200; // fixed for now

  u32 E  = task.exponent;

  // TODO: replace Saver with Pm1Saver which does not take the PRP stuff
  Saver saver{E, args.nSavefiles, args.startFrom, args.mprimeDir};

  auto [B1, k, data] = saver.loadP1();

  u32 desiredB1 = task.B1 ? task.B1 : args.B1;
  if (!B1) { B1 = desiredB1; }

  if (B1 != desiredB1) { log("using B1=%u (from savefile) vs. B1=%u\n", B1, desiredB1); }
  assert(B1);

  if (k == 0) {
    assert(data.empty());
    data = makeWords(E, 1);
  }

  auto powerBits = powerSmoothLE(E, B1, blockSize);
  const u32 nBits = powerBits.size();

  assert(nBits % blockSize == 0);
  assert(k % blockSize == 0); // can only save/verify P-1 at multiples of blockSize
  assert(k <= nBits);

  powerBits.resize(nBits - k); // drop the already processed bits

  writeData(data);
  // writeCheck(makeWords(E, 1u));
  writeCheck(data); // bufCheck << bufData;
  bufBase  << bufData;

  log("%5.2f%% @%u/%u B1(%u) %016" PRIx64 "\n", k*100.0f/nBits, k, nBits, B1, dataResidue());

  vector<bool> sumLE;
  Signal signal;
  bool updateCheck = false;
  optional<P1State> pendingSave;

  u32 lastTimerK = k;
  // u32 newTimerK = timerK;
  u32 startK = k;
  optional<u64> logRes;
  optional<bool> maybeOK = true;
  float checkSecs = 0;

  Timer timer;

  bool getOut = false;
  while (true) {
    if (powerBits.empty()) { getOut = true; }

    if (!getOut) {
      auto bits = takeTopBits(powerBits, blockSize);
      sumLE = addLE(sumLE, bits);

      pm1Block(bits, updateCheck); // here's GPU work
      updateCheck = true;
    }

    if (logRes) {
      u32 deltaIts = k - lastTimerK;
      assert(deltaIts % blockSize == 0);
      u32 nIts = deltaIts + deltaIts / blockSize;
      float secs = float(timer.reset()) - checkSecs;
      float secsPerIt = nIts ? secs / nIts : 0.0f;
      const char* strOK = maybeOK ? *maybeOK ? "K" : "E" : "";

      pm1Log(B1, k, nBits, strOK, *logRes, secsPerIt, checkSecs, nErr, readROE());
      lastTimerK = k;
    }

    if (pendingSave) {
      bool isDone = pendingSave->k >= nBits;
      saver.saveP1(*pendingSave, isDone);
      pendingSave.reset();
    }

    if (getOut) { break; }

    k += blockSize;

    bool resZero = logRes && *logRes == 0;
    logRes.reset();

    bool doStop  = signal.stopRequested();
    bool doCheck = resZero || doStop  || k % 40000 == 0 || k - startK == 2 * blockSize || powerBits.empty();
    bool doLog   = doCheck || k % 10000 == 0;

    if (!doCheck && !doLog) {
      finish();
      maybeOK.reset();
      continue;
    }

    if (doCheck) {
      data = readData();
      Timer checkTimer;
      if (data.empty()) { return RETRY; }
      logRes = residue(data);

      maybeOK = pm1Check(sumLE, blockSize);

      updateCheck = false;
      assert(maybeOK);
      checkSecs = checkTimer.at();

      if (maybeOK && *maybeOK) {
        assert(!pendingSave);
        pendingSave = P1State{.B1=B1, .k=k, .data=data};
      }

      if (!*maybeOK || doStop) { getOut = true; }

      logTimeKernels();
    } else {
      assert(doLog);
      logRes = dataResidue(); // implies finish()
      checkSecs = 0;
    }
  }

  assert(maybeOK);
  if (maybeOK && !*maybeOK) { return RETRY; }

  if (!powerBits.empty()) { throw "stop requested"; }

  log("completed\n");
  // auto factor = GCD(E, data, 1);
  // log("factor \"%s\"\n", factor.c_str());
  return DONE;
}

PRPResult Gpu::isPrimePRP(const Args &args, const Task& task) {
  u32 E = task.exponent;
  u32 k = 0, blockSize = 0;
  u32 nErrors = 0;
  
  u32 power = -1;
  u32 startK = 0;

  Saver saver{E, args.nSavefiles, args.startFrom, args.mprimeDir};
  Signal signal;

  // Used to detect a repetitive failure, which is more likely to indicate a software rather than a HW problem.
  std::optional<u64> lastFailedRes64;

  // Number of sequential errors (with no success in between). If this ever gets high enough, stop.
  int nSeqErrors = 0;
  
 reload:
  {
    PRPState loaded = saver.loadPRP(args.blockSize);    
    writeState(loaded.check, loaded.blockSize, buf1, buf2, buf3);
    
    u64 res = dataResidue();
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

  while (true) {
    assert(k < kEndEnd);
    
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

    bool doStop = false;

    if (k % blockSize == 0) {
      doStop = signal.stopRequested() || (args.iters && k - startK >= args.iters);
    }

    bool leadOut = doStop || (k % 10000 == 0) || (k % blockSize == 0 && k >= kEndEnd) || k == persistK || k == kEnd || useLongCarry;

    coreStep(bufData, bufData, leadIn, leadOut, false);
    leadIn = leadOut;    
    
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

    if (!leadOut) {
      if (k % blockSize == 0) { finish(); }
      continue;
    }

    u64 res = dataResidue(); // implies finish()
    bool doCheck = !res || doStop || (k % checkStep == 0) || (k >= kEndEnd) || (k - startK == 2 * blockSize);
      
    if (k % 10000 == 0 && !doCheck) {
      auto roeInfo = readROE();
      float secsPerIt = iterationTimer.reset(k);
      // log("   %9u %6.2f%% %s %4.0f us/it\n", k, k / float(kEndEnd) * 100, hex(res).c_str(), secsPerIt * 1'000'000);
      if (roeInfo.N) {
        log("%9u %s %4.0f; ROE=%.3f %.4f %u\n", k, hex(res).c_str(), secsPerIt * 1'000'000,
            roeInfo.max, roeInfo.norm, roeInfo.N);
      } else {
        log("%9u %s %4.0f\n", k, hex(res).c_str(), secsPerIt * 1'000'000);
      }
    }
      
    if (doStop) {
      log("Stopping, please wait..\n");
      signal.release();
    }
            
    if (doCheck) {
      // if (printStats) { printRoundoff(E); }

      float secsPerIt = iterationTimer.reset(k);

      Words check = readCheck();
      if (check.empty()) { log("Check read ZERO\n"); }

      bool ok = !check.empty() && this->doCheck(blockSize, buf1, buf2, buf3);

      float secsCheck = iterationTimer.reset(k);
        
      if (ok) {
        nSeqErrors = 0;
        lastFailedRes64.reset();
        skipNextCheckUpdate = true;

        if (k < kEnd) { saver.savePRP(PRPState{k, blockSize, res, check, nErrors}); }

        float secsSave = iterationTimer.reset(k);
          
        doBigLog(E, k, res, ok, secsPerIt, secsCheck, secsSave, kEndEnd, nErrors);
          
        if (k >= kEndEnd) {
          fs::path proofFile = saveProof(args, proofSet);
          return {"", isPrime, finalRes64, nErrors, proofFile.string()};          
        }        
      } else {
        doBigLog(E, k, res, ok, secsPerIt, secsCheck, 0, kEndEnd, nErrors);
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
        queue->finish();
        throw "stop requested";
      }
        
      iterationTimer.reset(k);
    }
  }
}

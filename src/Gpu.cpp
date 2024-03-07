// Copyright Mihai Preda and George Woltman.

#include "Gpu.h"
#include "Proof.h"
#include "state.h"
#include "Args.h"
#include "Signal.h"
#include "FFTConfig.h"
#include "Queue.h"
#include "Task.h"
#include "Memlock.h"
#include "KernelCompiler.h"
#include "SaveMan.h"
#include "timeutil.h"

#define _USE_MATH_DEFINES
#include <cmath>

#include <cstring>
#include <algorithm>
#include <future>
#include <optional>
#include <bitset>
#include <limits>
#include <iomanip>
#include <array>
#include <cinttypes>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static_assert(sizeof(double2) == 16, "size double2");
static_assert(sizeof(long double) > sizeof(double), "long double offers extended precision");

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
    
#if 0
    auto angle = - M_PIl * k / N;
    return {cosl(angle), sinl(angle)};
#else
    double angle = - M_PIl * k / N;
    return {cos(angle), sin(angle)};
#endif
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

#if 1
  for (u32 line = 1; line < radix; ++line) {
    for (u32 col = 0; col < size / radix; ++col) {
      tab.push_back(root1<double>(size, col * line));
    }
  }
  tab.resize(size);
#else
  tab.resize(size);
  auto *p = tab.data() + radix;
  for (u32 w = radix; w < size; w *= radix) { p = smallTrigBlock(w, std::min(radix, size / w), p); }
  assert(p - tab.data() == size);
#endif

  return {context, "smallTrig", tab};
}

ConstBuffer<double2> genMiddleTrig(const Context& context, u32 smallH, u32 middle) {
  vector<double2> tab;
  if (middle == 1) {
    tab.resize(1);
  } else {  
    u32 size = smallH * (middle - 1);
    tab.resize(size);
    [[maybe_unused]] auto *p = smallTrigBlock(smallH, middle, tab.data());
    assert(p - tab.data() == size);
  }
  return {context, "middleTrig", tab};
}

template<typename T>
vector<pair<T, T>> makeTrig(u32 n, vector<pair<T,T>> tab = {}) {
  assert(n % 8 == 0);
  tab.reserve(tab.size() + n/8 + 1);
  for (u32 k = 0; k <= n/8; ++k) { tab.push_back(root1<T>(n, k)); }
  return tab;
}

template<typename T>
vector<pair<T, T>> makeTinyTrig(u32 W, u32 hN, vector<pair<T, T>> tab = {}) {
  tab.reserve(tab.size() + W/2 + 1);
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

string toLiteral(const string& s) { return s; }

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

string clArgs(cl_device_id id, u32 N, u32 E, u32 WIDTH, u32 SMALL_HEIGHT, u32 MIDDLE, u32 nW) {
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
  string s;
  for (const auto& d : defines) {
    s += " -D"s + string(d);
  }
  return s;
}

string clArgs(const Args& args) {
  string s;
  for (const auto& [key, val] : args.flags) {
    s += " -D" + key + '=' + toLiteral(val);
  }
  return s;
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
  queue(Queue::make(context, timeKernels, args.cudaYield)),
  
#define K(name, ...) name(#name, queue, __VA_ARGS__)

  //  W / nW
  K(kernCarryFused,    "carryfused.cl", "carryFused", W * (BIG_H + 1) / nW),
  K(kernCarryFusedMul, "carryfused.cl", "carryFused", W * (BIG_H + 1) / nW, "-DMUL3=1"),
  K(carryFusedLL,      "carryfused.cl", "carryFused", W * (BIG_H + 1) / nW, "-DLL=1"),

  K(kernCarryA, "carry.cl", "carry", hN / CARRY_LEN),
  K(kernCarryM, "carry.cl", "carry", hN / CARRY_LEN, "-DMUL3=1"),
  K(carryLL,    "carry.cl", "carry", hN / CARRY_LEN, "-DLL=1"),
  K(carryB, "carryb.cl", "carryB",   hN / CARRY_LEN),

  K(fftP, "fftp.cl", "fftP", hN / nW),
  K(fftW, "fftw.cl", "fftW", hN / nW),
  
  // SMALL_H / nH
  K(fftHin,  "ffthin.cl",  "fftHin",  hN / nH),
  K(fftHout, "ffthout.cl", "fftHout", hN / nH),

  K(tailSquare,    "tailsquare.cl", "tailSquare", hN / nH / 2),
  K(tailSquareLow, "tailsquare.cl", "tailSquare", hN / nH / 2, "-DMUL_LOW=1"),
  K(tailMul,       "tailmul.cl", "tailMul",       hN / nH / 2),
  K(tailMulLow,    "tailmul.cl", "tailMul",       hN / nH / 2, "-DMUL_LOW=1"),
  
  // 256
  K(fftMiddleIn,  "fftmiddlein.cl",  "fftMiddleIn",  hN / (BIG_H / SMALL_H)),
  K(fftMiddleOut, "fftmiddleout.cl", "fftMiddleOut", hN / (BIG_H / SMALL_H)),
  
  // 64
  K(transposeIn,  "transpose.cl", "transposeIn",  hN / 64),
  K(transposeOut, "transpose.cl", "transposeOut", hN / 64),
  
  K(readResidue, "etc.cl", "readResidue", 64),

  // 256
  K(isNotZero, "etc.cl", "isNotZero", 256 * 256),
  K(isEqual,   "etc.cl", "isEqual",   256 * 256),
  K(sum64,     "etc.cl", "sum64",     256 * 256),
#undef K

  bufTrigW{genSmallTrig(context, W, nW)},
  bufTrigH{genSmallTrig(context, SMALL_H, nH)},
  bufTrigM{genMiddleTrig(context, SMALL_H, BIG_H / SMALL_H)},
  
  bufTrigBHW{context, "trigBHW", makeTinyTrig(W, hN, makeTrig<double>(BIG_H))},
  bufTrig2SH{context, "trig2SH", makeTrig<double>(2 * SMALL_H)},
  bufThreadWeights{context, "threadWeights", weights.threadWeightsIF},
  bufCarryWeights{context,  "carryWeights",  weights.carryWeightsIF},
  
  bufBits{context, "bits", weights.bitsCF},
  bufBitsC{context, "bitsC", weights.bitsC},
  bufData{queue, "data", N},
  bufAux{queue, "aux", N},
  bufCheck{queue, "check", N},
  bufBase{queue, "base", N},
  bufCarry{queue, "carry", N / 2},
  bufReady{queue, "ready", BIG_H},
  bufSmallOut{queue, "smallOut", 256},
  bufSumOut{queue, "sumOut", 1},
  bufROE{queue, "ROE", ROE_SIZE},
  roePos{0},
  buf1{queue, "buf1", N},
  buf2{queue, "buf2", N},
  buf3{queue, "buf3", N},
  statsBits{u32(args.value("STATS", 0))},
  args{args}
{
  log("Stats: %x\n", statsBits);
  string commonArgs = clArgs(device, N, E, W, SMALL_H, BIG_H / SMALL_H, nW) + clArgs(args);
  
  KernelCompiler compiler{args.cacheDir.string().c_str(), context.get(), device, commonArgs, args.dump};
  for (Kernel* k : {&kernCarryFused, &kernCarryFusedMul, &carryFusedLL,
       &fftP, &fftW, &fftHin, &fftHout,
       &fftMiddleIn, &fftMiddleOut, &kernCarryA, &kernCarryM, &carryLL, &carryB,
       &transposeIn, &transposeOut,
       &tailMulLow, &tailMul, &tailSquare, &tailSquareLow,
       &readResidue, &isNotZero, &isEqual, &sum64}) {
    k->load(compiler, device);
  }
  
  for (Kernel* k : {&kernCarryFused, &kernCarryFusedMul, &carryFusedLL}) {
    k->setFixedArgs(3, bufCarry, bufReady, bufTrigW, bufBits, bufROE, bufThreadWeights, bufCarryWeights);
  }

  fftP.setFixedArgs(2, bufTrigW, bufThreadWeights, bufCarryWeights);
  fftW.setFixedArgs(2, bufTrigW);
  fftHin.setFixedArgs(2, bufTrigH);
  fftHout.setFixedArgs(1, bufTrigH);
  
  fftMiddleIn.setFixedArgs( 2, bufTrigM, bufTrigBHW);
  fftMiddleOut.setFixedArgs(2, bufTrigM, bufTrigBHW);
  
  for (Kernel* k : {&kernCarryA, &kernCarryM, &carryLL}) {
    k->setFixedArgs(3, bufCarry, bufBitsC, bufROE, bufThreadWeights, bufCarryWeights);
  }

  carryB.setFixedArgs(1, bufCarry, bufBitsC);
  tailMulLow.setFixedArgs(3, bufTrigH, bufTrig2SH, bufTrigBHW);
  tailMul.setFixedArgs(3, bufTrigH, bufTrig2SH, bufTrigBHW);
  tailSquare.setFixedArgs(2, bufTrigH, bufTrig2SH, bufTrigBHW);
  tailSquareLow.setFixedArgs(2, bufTrigH, bufTrig2SH, bufTrigBHW);

  bufReady.zero();
  bufROE.zero();
  finish();
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

ROEInfo Gpu::readROE() {
  assert(roePos <= ROE_SIZE);
  if (roePos) {
    vector<float> roe = bufROE.read(roePos);
    assert(roe.size() == roePos);
    bufROE.zero(roePos);
    roePos = 0;

    double sumRoe = 0;
    double sum2Roe = 0;
    float maxRoe = 0;

#if DUMP_ROE
    File froe = File::openAppend("roe.txt");
#endif

    for (auto x : roe) {
      assert(x >= 0);

#if DUMP_ROE
      froe.printf("%f\n", x);
#endif

      maxRoe = max(x, maxRoe);
      sumRoe  += x;
      sum2Roe += x * x;
    }
    u32 n = roe.size();
    float invN = 1.0f / n;

    float sdRoe = sqrtf(n * sum2Roe - sumRoe * sumRoe) * invN;
    float meanRoe = float(sumRoe) * invN;

    return {n, {maxRoe, meanRoe, sdRoe}};
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

template<typename T>
static bool isAllZero(vector<T> v) { return std::all_of(v.begin(), v.end(), [](T x) { return x == 0;}); }

vector<u32> Gpu::readAndCompress(ConstBuffer<int>& buf)  {
  for (int nRetry = 0; nRetry < 3; ++nRetry) {
    sum64(bufSumOut, u32(buf.size * sizeof(int)), buf);
    
    vector<u64> expectedVect(1);
    bufSumOut.readAsync(expectedVect);
    vector<int> data = readOut(buf);
    u64 gpuSum = expectedVect[0];
    
    u64 sum = 0;
    for (auto it = data.begin(), end = data.end(); it < end; it += 2) {
      sum += u32(*it) | (u64(*(it + 1)) << 32);
    }

    if (sum != gpuSum) {
      log("GPU read failed: %016" PRIx64 " (gpu) != %016" PRIx64 " (host)\n", gpuSum, sum);
      continue; // have another try
    }

    if (gpuSum == 0 && isAllZero(data)) {
      log("Read all-zero\n");
      return {};
    }

    return compactBits(std::move(data), E);
  }
  throw "Persistent read errors: GPU->Host";
}

vector<u32> Gpu::readCheck() { return readAndCompress(bufCheck); }
vector<u32> Gpu::readData() { return readAndCompress(bufData); }

// out := inA * inB;
void Gpu::mul(Buffer<int>& out, Buffer<int>& inA, Buffer<double>& inB, Buffer<double>& tmp1, Buffer<double>& tmp2, bool mul3) {
    fftP(tmp1, inA);
    fftMiddleIn(tmp2, tmp1);
    tailMul(tmp1, inB, tmp2);
    fftMiddleOut(tmp2, tmp1);
    fftW(tmp1, tmp2);
    if (mul3) { carryM(out, tmp1); } else { carryA(out, tmp1); }
    carryB(out);
}

// out := inA * inB;
void Gpu::modMul(Buffer<int>& out, Buffer<int>& inA, Buffer<int>& inB, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3, bool mul3) {

  fftP(buf1, inB);
  fftMiddleIn(buf3, buf1);
  
  mul(out, inA, buf3, buf1, buf2, mul3);
};

void Gpu::mul(Buffer<int>& io, Buffer<int>& inA, Buffer<int>& inB) { modMul(io, inA, inB, buf1, buf2, buf3); }

void Gpu::mul(Buffer<int>& io, Buffer<int>& inB) { mul(io, io, inB); }

void Gpu::mul(Buffer<int>& io, Buffer<double>& buf1) {
  // We know that coreStep() stores double output in buf1; so we're going to use buf2 & buf3 for temps.
  mul(io, io, buf1, buf2, buf3, false);
}

void Gpu::writeState(const vector<u32> &check, u32 blockSize, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
  assert(blockSize > 0);
  writeCheck(check);
  bufData << bufCheck;
  bufAux  << bufCheck;
  
  u32 n = 0;
  for (n = 1; blockSize % (2 * n) == 0; n *= 2) {
    squareLoop(bufData, 0, n);
    modMul(bufData, bufData, bufAux, buf1, buf2, buf3);
    bufAux << bufData;
  }
  
  assert((n & (n - 1)) == 0);
  assert(blockSize % n == 0);
    
  blockSize /= n;
  assert(blockSize >= 2);
  
  for (u32 i = 0; i < blockSize - 2; ++i) {
    squareLoop(bufData, 0, n);
    modMul(bufData, bufData, bufAux, buf1, buf2, buf3);
  }
  
  squareLoop(bufData, 0, n);
  modMul(bufData, bufData, bufAux, buf1, buf2, buf3, true);
}
  
bool Gpu::doCheck(u32 blockSize, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
  squareLoop(bufAux, bufCheck, 0, blockSize, true);
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
    squareLoop(bufData, 0, its);
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
    fftMiddleIn(buf3, buf2);
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
  fftMiddleIn(out, tmp1);
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
  fftMiddleOut(out, tmp);
  
  int p = 63;
  while (!testBit(exp, p)) { --p; }
  
  for (--p; ; --p) {
    if (testBit(exp, p)) {
      doCarry(tmp, out);
      fftMiddleIn(out, tmp);
      tailMulLow(tmp, out, base);
      fftMiddleOut(out, tmp);
    }
    
    if (!p) { break; }

    doCarry(tmp, out);
    fftMiddleIn(out, tmp);
    tailSquare(tmp, out);
    fftMiddleOut(out, tmp);
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

void Gpu::square(Buffer<int>& out, Buffer<int>& in, bool leadIn, bool leadOut, bool mul3) {
  if (leadIn) {
    fftP(buf2, in);
    fftMiddleIn(buf1, buf2);
  }
  
  tailSquare(buf2, buf1);
  
  fftMiddleOut(buf1, buf2);

  if (leadOut) {
    fftW(buf2, buf1);
    if (mul3) { carryM(out, buf2); } else { carryA(out, buf2); }
    carryB(out);
  } else {
    assert(!useLongCarry); // because useLongCarry always sets leadOut=true
    if (mul3) { carryFusedMul(buf2, buf1); } else { carryFused(buf2, buf1); }
    fftMiddleIn(buf1, buf2);
  }
}

u32 Gpu::squareLoop(Buffer<int>& out, Buffer<int>& in, u32 from, u32 to, bool doTailMul3) {
  assert(from < to);
  bool leadIn = true;
  for (u32 k = from; k < to; ++k) {
    bool leadOut = useLongCarry || (k == to - 1);
    square(out, (k==from) ? in : out, leadIn, leadOut, doTailMul3 && (k == to - 1));
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

PRPResult Gpu::isPrimePRP(const Args &args, const Task& task) {
  u32 E = task.exponent;
  u32 k = 0, blockSize = 0;
  u32 nErrors = 0;
  
  u32 power = -1;
  u32 startK = 0;

  StateSaver<PRPState> saver{E};
  Signal signal;

  // Used to detect a repetitive failure, which is more likely to indicate a software rather than a HW problem.
  std::optional<u64> lastFailedRes64;

  // Number of sequential errors (with no success in between). If this ever gets high enough, stop.
  int nSeqErrors = 0;
  
 reload:
  {
    PRPState loaded = saver.load();
        
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
    power = ProofSet::effectivePower(E, args.proofPow, startK);
    
    if (power != args.proofPow) {
      log("Proof using power %u (vs %u)\n", power, args.proofPow);
    }
    
    if (!power) {
      log("Proof generation disabled\n");
    } else {
      if (power > ProofSet::bestPower(E)) {
        log("Warning: proof power %u is excessively large; use at most power %u\n",
            power, ProofSet::bestPower(E));
      }

      log("Proof of power %u requires about %.1fGB of disk space\n",
          power, ProofSet::diskUsageGB(E, power));
    }
  }
  
  ProofSet proofSet{E, power};

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

    square(bufData, leadIn, leadOut, false);
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

      if (roeInfo.N) {
        Stats &roe = roeInfo.roe;
        log("%9u %s %4.0f; %s %.3f N=%u z=%.1f\n",
            k, hex(res).c_str(), secsPerIt * 1'000'000,
            (statsBits & 0x10 ) ? "Carry" : "ROE",
            roe.max, roeInfo.N, roe.z(.5f));
      } else {
        log("%9u %s %4.0f\n",
            k, hex(res).c_str(), secsPerIt * 1'000'000);
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
      bool ok = !check.empty() && this->doCheck(blockSize, buf1, buf2, buf3);

      float secsCheck = iterationTimer.reset(k);
        
      if (ok) {
        nSeqErrors = 0;
        lastFailedRes64.reset();
        skipNextCheckUpdate = true;

        if (k < kEnd) { saver.save({E, k, blockSize, res, check, nErrors}); }

        float secsSave = iterationTimer.reset(k);
          
        doBigLog(E, k, res, ok, secsPerIt, secsCheck, secsSave, kEndEnd, nErrors);
          
        if (k >= kEndEnd) {
          fs::path proofFile = saveProof(args, proofSet);
          return {isPrime, finalRes64, nErrors, proofFile.string()};
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

PRPResult Gpu::isPrimeLL(const Args& args, const Task& task) {
  u32 E = task.exponent;

  StateSaver<LLState> saver{E};
  Signal signal;

  u32 startK = 0;
  {
    LLState state = saver.load();

    startK = state.k;
    writeData(state.data);
    u64 expectedRes = (u64(state.data[1]) << 32) | state.data[0];
    u64 res = dataResidue();
    assert(res == expectedRes);
    log("LL loaded @ %u : %016" PRIx64 "\n", startK, res);
  }

  u32 k = startK;
  u32 kEnd = E - 2;
  bool leadIn = true;

  while (k < kEnd) {
    if (signal.stopRequested()) {
      // TODO
    }

    // coreStep(bufData, bufData, leadIn, leadOut, false);

  }




  return {}; //TODO
}

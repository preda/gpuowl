// Copyright (C) Mihai Preda and George Woltman.

#include "Gpu.h"
#include "Proof.h"
#include "TimeInfo.h"
#include "Trig.h"
#include "state.h"
#include "Args.h"
#include "Signal.h"
#include "FFTConfig.h"
#include "Queue.h"
#include "Task.h"
#include "KernelCompiler.h"
#include "Saver.h"
#include "timeutil.h"
#include "TrigBufCache.h"
#include "fs.h"
#include "Sha3Hash.h"

#include <algorithm>
#include <bitset>
#include <limits>
#include <iomanip>
#include <array>
#include <cinttypes>

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

namespace {

u32 kAt(u32 H, u32 line, u32 col) { return (line + col * H) * 2; }

double weight(u32 N, u32 E, u32 H, u32 line, u32 col, u32 rep) {
  return exp2l((long double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

double invWeight(u32 N, u32 E, u32 H, u32 line, u32 col, u32 rep) {
  return exp2l(-(long double)(extra(N, E, kAt(H, line, col) + rep)) / N);
}

double weightM1(u32 N, u32 E, u32 H, u32 line, u32 col, u32 rep) {
  return exp2l((long double)(extra(N, E, kAt(H, line, col) + rep)) / N) - 1;
}

double invWeightM1(u32 N, u32 E, u32 H, u32 line, u32 col, u32 rep) {
  return exp2l(- (long double)(extra(N, E, kAt(H, line, col) + rep)) / N) - 1;
}

double boundUnderOne(double x) { return std::min(x, nexttoward(1, 0)); }

#define CARRY_LEN 8

Weights genWeights(u32 E, u32 W, u32 H, u32 nW, bool AmdGpu) {
  u32 N = 2u * W * H;
  
  u32 groupWidth = W / nW;

  // Inverse + Forward
  vector<double> weightsConstIF;
  vector<double> weightsIF;
  for (u32 thread = 0; thread < groupWidth; ++thread) {
    auto iw = invWeight(N, E, H, 0, thread, 0);
    auto w = weight(N, E, H, 0, thread, 0);
    // nVidia GPUs have a constant cache that only works on buffer sizes less than 64KB.  Create a smaller buffer
    // that is a copy of the first part of weightsIF.  There are several kernels that need the combined weightsIF
    // buffer, so there is an unfortunate duplication of these weights.
    if (!AmdGpu) {
      weightsConstIF.push_back(2 * boundUnderOne(iw));
      weightsConstIF.push_back(2 * w);
    }
    weightsIF.push_back(2 * boundUnderOne(iw));
    weightsIF.push_back(2 * w);
  }

  // the group order matches CarryA/M (not fftP/CarryFused).
  for (u32 gy = 0; gy < H; ++gy) {
    weightsIF.push_back(invWeightM1(N, E, H, gy, 0, 0));
    weightsIF.push_back(weightM1(N, E, H, gy, 0, 0));
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

  return Weights{weightsConstIF, weightsIF, bits, bitsC};
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

template<typename T, size_t N>
string toLiteral(const std::array<T, N>& v) {
  string s = "{";
  for (T x : v) {
    s += toLiteral(x) + ",";
  }
  s += "}";
  return s;
}

string toLiteral(const string& s) { return s; }

string toLiteral(double2 cs) { return "U2("s + toLiteral(cs.first) + ',' + toLiteral(cs.second) + ')'; }

template<typename T>
string toDefine(const string& k, T v) { return " -D"s + k + '=' + toLiteral(v); }

template<typename T>
string toDefine(const T& vect) {
  string s;
  for (const auto& [k, v] : vect) { s += toDefine(k, v); }
  return s;
}

constexpr bool isInList(const string& s, initializer_list<string> list) {
  for (const string& e : list) { if (e == s) { return true; }}
  return false;
}

string clDefines(const Args& args, cl_device_id id, FFTConfig fft, const vector<KeyVal>& extraConf, u32 E, bool doLog,
                 bool &tail_single_wide, bool &tail_single_kernel, u32 &tail_trigs) {
  map<string, string> config;

  // Highest priority is the requested "extra" conf
  config.insert(extraConf.begin(), extraConf.end());

  // Next, args config
  config.insert(args.flags.begin(), args.flags.end());

  // Lowest priority: the per-FFT config if any
  if (auto it = args.perFftConfig.find(fft.shape.spec()); it != args.perFftConfig.end()) {
    // log("Found %s\n", fft.shape.spec().c_str());
    config.insert(it->second.begin(), it->second.end());
  }

  // Default value for -use options that must also be parsed in C++ code
  tail_single_wide = 0, tail_single_kernel = 0;         // Default tailSquare is double-wide with two kernels
  tail_trigs = 2;                                       // Default to calculating from scratch, no memory accesses

  // Validate -use options
  for (const auto& [k, v] : config) {
    bool isValid = isInList(k, {
                              "FAST_BARRIER",
                              "STATS",
                              "IN_SIZEX",
                              "IN_WG",
                              "OUT_SIZEX",
                              "OUT_WG",
                              "UNROLL_H",
                              "UNROLL_W",
                              "NO_ASM",
                              "DEBUG",
                              "CARRY64",
                              "BCAST",
                              "BIGLIT",
                              "NONTEMPORAL",
                              "PAD",
                              "TAIL_KERNELS",
                              "TAIL_TRIGS"
                            });
    if (!isValid) {
      log("Warning: unrecognized -use key '%s'\n", k.c_str());
    }

    // Some -use options are needed in both OpenCL code and C++ initialization code
    if (k == "TAIL_KERNELS") {
      if (atoi(v.c_str()) == 0) tail_single_wide = 1, tail_single_kernel = 1;
      if (atoi(v.c_str()) == 1) tail_single_wide = 1, tail_single_kernel = 0;
      if (atoi(v.c_str()) == 2) tail_single_wide = 0, tail_single_kernel = 1;
      if (atoi(v.c_str()) == 3) tail_single_wide = 0, tail_single_kernel = 0;
    }
    if (k == "TAIL_TRIGS") tail_trigs = atoi(v.c_str());
  }

  string defines = toDefine(config);
  if (doLog) { log("config: %s\n", defines.c_str()); }

  defines += toDefine(initializer_list<pair<string, u32>>{
                    {"EXP", E},
                    {"WIDTH", fft.shape.width},
                    {"SMALL_HEIGHT", fft.shape.height},
                    {"MIDDLE", fft.shape.middle},
                    {"CARRY_LEN", CARRY_LEN},
                    {"NW", fft.shape.nW()},
                    {"NH", fft.shape.nH()}
                  });

  if (isAmdGpu(id)) { defines += toDefine("AMDGPU", 1); }

  if ((fft.carry == CARRY_AUTO && fft.shape.needsLargeCarry(E)) || (fft.carry == CARRY_64)) {
    if (doLog) { log("Using CARRY64\n"); }
    defines += toDefine("CARRY64", 1);
  }

  u32 N = fft.shape.size();

  defines += toDefine("WEIGHT_STEP", weightM1(N, E, fft.shape.height * fft.shape.middle, 0, 0, 1));
  defines += toDefine("IWEIGHT_STEP", invWeightM1(N, E, fft.shape.height * fft.shape.middle, 0, 0, 1));
  defines += toDefine("FFT_VARIANT", fft.variant);
  defines += toDefine("TAILT", root1Fancy(fft.shape.height * 2, 1));

  TrigCoefs coefs = trigCoefs(fft.shape.size() / 4);
  defines += toDefine("TRIG_SCALE", int(coefs.scale));
  defines += toDefine("TRIG_SIN",  coefs.sinCoefs);
  defines += toDefine("TRIG_COS",  coefs.cosCoefs);

  // Calculate fractional bits-per-word = (E % N) / N * 2^64
  u32 bpw_hi = (u64(E % N) << 32) / N;
  u32 bpw_lo = (((u64(E % N) << 32) % N) << 32) / N;
  u64 bpw = (u64(bpw_hi) << 32) + bpw_lo;
  bpw--; // bpw must not be an exact value -- it must be less than exact value to get last biglit value right
  defines += toDefine("FRAC_BPW_HI", (u32) (bpw >> 32));
  defines += toDefine("FRAC_BPW_LO", (u32) bpw);
  u32 bigstep = (bpw * (N / fft.shape.nW())) >> 32;
  defines += toDefine("FRAC_BITS_BIGSTEP", bigstep);

  return defines;
}

template<typename T>
pair<vector<T>, vector<T>> split(const vector<T>& v, const vector<u32>& select) {
  vector<T> a;
  vector<T> b;
  auto selIt = select.begin();
  u32 selNext = selIt == select.end() ? u32(-1) : *selIt;
  for (u32 i = 0; i < v.size(); ++i) {
    if (i == selNext) {
      b.push_back(v[i]);
      ++selIt;
      selNext = selIt == select.end() ? u32(-1) : *selIt;
    } else {
      a.push_back(v[i]);
    }
  }
  return {a, b};
}

RoeInfo roeStat(const vector<float>& roe) {
  double sumRoe = 0;
  double sum2Roe = 0;
  double maxRoe = 0;

  for (auto xf : roe) {
    double x = xf;
    assert(x >= 0);
    maxRoe = max(x, maxRoe);
    sumRoe  += x;
    sum2Roe += x * x;
  }
  u32 n = roe.size();

  double sdRoe = sqrt(n * sum2Roe - sumRoe * sumRoe) / n;
  double meanRoe = sumRoe / n;

  return {n, maxRoe, meanRoe, sdRoe};
}

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

u32 baseCheckStep(u32 blockSize) {
  switch (blockSize) {
    case 200:  return 40'000;
    case 400:  return 160'000;
    case 500:  return 200'000;
    case 1000: return 1'000'000;
    default:
      assert(false);
      return 0;
  }
}

u32 checkStepForErrors(u32 blockSize, u32 nErrors) {
  u32 step = baseCheckStep(blockSize);
  return nErrors ? step / 2 : step;
}

string toHex(u32 x) {
  char buf[16];
  snprintf(buf, sizeof(buf), "%08x", x);
  return buf;
}

string toHex(const vector<u32>& v) {
  string s;
  for (auto it = v.rbegin(), end = v.rend(); it != end; ++it) {
    s += toHex(*it);
  }
  return s;
}

} // namespace

// --------

unique_ptr<Gpu> Gpu::make(Queue* q, u32 E, GpuCommon shared, FFTConfig fftConfig, const vector<KeyVal>& extraConf, bool logFftSize) {
  return make_unique<Gpu>(q, shared, fftConfig, E, extraConf, logFftSize);
}

Gpu::~Gpu() {
  // Background tasks may have captured *this*, so wait until those are complete before destruction
  background->waitEmpty();
}

#define ROE_SIZE 100000
#define CARRY_SIZE 100000

Gpu::Gpu(Queue* q, GpuCommon shared, FFTConfig fft, u32 E, const vector<KeyVal>& extraConf, bool logFftSize) :
  queue(q),
  background{shared.background},
  args{*shared.args},
  E(E),
  N(fft.shape.size()),
  WIDTH(fft.shape.width),
  SMALL_H(fft.shape.height),
  BIG_H(SMALL_H * fft.shape.middle),
  hN(N / 2),
  nW(fft.shape.nW()),
  nH(fft.shape.nH()),
  bufSize(N * sizeof(double)),
  useLongCarry{args.carry == Args::CARRY_LONG},
  compiler{args, queue->context, clDefines(args, queue->context->deviceId(), fft, extraConf, E, logFftSize, tail_single_wide, tail_single_kernel, tail_trigs)},

#define K(name, ...) name(#name, &compiler, profile.make(#name), queue, __VA_ARGS__)

  //  W / nW
  K(kCarryFused,    "carryfused.cl", "carryFused", WIDTH * (BIG_H + 1) / nW),
  K(kCarryFusedROE, "carryfused.cl", "carryFused", WIDTH * (BIG_H + 1) / nW, "-DROE=1"),

  K(kCarryFusedMul,    "carryfused.cl", "carryFused", WIDTH * (BIG_H + 1) / nW, "-DMUL3=1"),
  K(kCarryFusedMulROE, "carryfused.cl", "carryFused", WIDTH * (BIG_H + 1) / nW, "-DMUL3=1 -DROE=1"),

  K(kCarryFusedLL,     "carryfused.cl", "carryFused", WIDTH * (BIG_H + 1) / nW, "-DLL=1"),

  K(kCarryA,    "carry.cl", "carry", hN / CARRY_LEN),
  K(kCarryAROE, "carry.cl", "carry", hN / CARRY_LEN, "-DROE=1"),

  K(kCarryM,    "carry.cl", "carry", hN / CARRY_LEN, "-DMUL3=1"),
  K(kCarryMROE, "carry.cl", "carry", hN / CARRY_LEN, "-DMUL3=1 -DROE=1"),

  K(kCarryLL,   "carry.cl", "carry", hN / CARRY_LEN, "-DLL=1"),
  K(carryB, "carryb.cl", "carryB",   hN / CARRY_LEN),

  K(fftP, "fftp.cl", "fftP", hN / nW),
  K(fftW, "fftw.cl", "fftW", hN / nW),
  
  // SMALL_H / nH
  K(fftHin,  "ffthin.cl",  "fftHin",  hN / nH),
  K(tailSquareZero, "tailsquare.cl", "tailSquareZero", SMALL_H / nH * 2),
  K(tailSquare, "tailsquare.cl", "tailSquare", !tail_single_wide && !tail_single_kernel ? hN / nH - SMALL_H / nH * 2 : // Double-wide tailSquare with two kernels
                                               !tail_single_wide ? hN / nH :                                           // Double-wide tailSquare with one kernel
                                               !tail_single_kernel ? hN / nH / 2 - SMALL_H / nH :                      // Single-wide tailSquare with two kernels
                                               hN / nH / 2),                                                           // Single-wide tailSquare with one kernel

  K(tailMul,       "tailmul.cl", "tailMul",       hN / nH / 2),
  K(tailMulLow,    "tailmul.cl", "tailMul",       hN / nH / 2, "-DMUL_LOW=1"),
  
  // 256
  K(fftMidIn,  "fftmiddlein.cl",  "fftMiddleIn",  hN / (BIG_H / SMALL_H)),
  K(fftMidOut, "fftmiddleout.cl", "fftMiddleOut", hN / (BIG_H / SMALL_H)),
  
  // 64
  K(transpIn,  "transpose.cl", "transposeIn",  hN / 64),
  K(transpOut, "transpose.cl", "transposeOut", hN / 64),
  
  K(readResidue, "etc.cl", "readResidue", 32, "-DREADRESIDUE=1"),

  // 256
  K(kernIsEqual, "etc.cl", "isEqual", 256 * 256, "-DISEQUAL=1"),
  K(sum64,       "etc.cl", "sum64",   256 * 256, "-DSUM64=1"),
  K(testTrig,    "selftest.cl", "testTrig", 256 * 256),
  K(testFFT4, "selftest.cl", "testFFT4", 256),
  K(testFFT, "selftest.cl", "testFFT", 256),
  K(testFFT15, "selftest.cl", "testFFT15", 256),
  K(testFFT14, "selftest.cl", "testFFT14", 256),
  K(testTime, "selftest.cl", "testTime", 4096 * 64),
#undef K

  bufTrigW{shared.bufCache->smallTrig(WIDTH, nW)},
  bufTrigH{shared.bufCache->smallTrigCombo(WIDTH, fft.shape.middle, SMALL_H, nH, fft.variant, tail_single_wide, tail_trigs)},
  bufTrigM{shared.bufCache->middleTrig(SMALL_H, BIG_H / SMALL_H, WIDTH)},

  weights{genWeights(E, WIDTH, BIG_H, nW, isAmdGpu(q->context->deviceId()))},

  bufConstWeights{q->context, std::move(weights.weightsConstIF)},
  bufWeights{q->context,      std::move(weights.weightsIF)},
  bufBits{q->context,         std::move(weights.bitsCF)},
  bufBitsC{q->context,        std::move(weights.bitsC)},

#define BUF(name, ...) name{profile.make(#name), queue, __VA_ARGS__}

  BUF(bufData, N),
  BUF(bufAux, N),

  BUF(bufCheck, N),
  BUF(bufBase, N),
  // Every double-word (i.e. N/2) produces one carry. In addition we may have one extra group thus WIDTH more carries.
  BUF(bufCarry,  N / 2 + WIDTH),
  BUF(bufReady, (N / 2 + WIDTH) / 32), // Every wavefront (32 or 64 lanes) needs to signal "carry is ready"

  BUF(bufSmallOut, 256),
  BUF(bufSumOut,     1),
  BUF(bufTrue,       1),
  BUF(bufROE, ROE_SIZE),
  BUF(bufStatsCarry, CARRY_SIZE),

  BUF(buf1, N + N/4),           // Let's us play with padding instead of rotating.  Need to calculate actual cost of padding
  BUF(buf2, N + N/4),
  BUF(buf3, N + N/4),
#undef BUF

  statsBits{u32(args.value("STATS", 0))},
  timeBufVect{profile.make("proofBufVect")}
{    

  float bitsPerWord = E / float(N);
  if (logFftSize) {
    log("FFT: %s %s (%.2f bpw)\n", numberK(N).c_str(), fft.spec().c_str(), bitsPerWord);

    // Sometimes we do want to run a FFT beyond a reasonable BPW (e.g. during -ztune), and these situations
    // coincide with logFftSize == false
    if (fft.maxExp() < E) {
      log("Warning: %s (max %u) may be too small for %u\n", fft.spec().c_str(), fft.maxExp(), E);
    }
  }

  if (bitsPerWord < FFTShape::MIN_BPW) {
    log("FFT size too large for exponent (%.2f bits/word < %.2f bits/word).\n", bitsPerWord, FFTShape::MIN_BPW);
    throw "FFT size too large";
  }

  useLongCarry = useLongCarry || (bitsPerWord < 12.0);

  if (useLongCarry) { log("Using long carry!\n"); }
  
  for (Kernel* k : {&kCarryFused, &kCarryFusedROE, &kCarryFusedMul, &kCarryFusedMulROE, &kCarryFusedLL}) {
    k->setFixedArgs(3, bufCarry, bufReady, bufTrigW, bufBits, bufConstWeights, bufWeights);
  }

  for (Kernel* k : {&kCarryFusedROE, &kCarryFusedMulROE})           { k->setFixedArgs(9, bufROE); }
  for (Kernel* k : {&kCarryFused, &kCarryFusedMul, &kCarryFusedLL}) { k->setFixedArgs(9, bufStatsCarry); }

  for (Kernel* k : {&kCarryA, &kCarryAROE, &kCarryM, &kCarryMROE, &kCarryLL}) {
    k->setFixedArgs(3, bufCarry, bufBitsC, bufWeights);
  }

  for (Kernel* k : {&kCarryAROE, &kCarryMROE})      { k->setFixedArgs(6, bufROE); }
  for (Kernel* k : {&kCarryA, &kCarryM, &kCarryLL}) { k->setFixedArgs(6, bufStatsCarry); }

  fftP.setFixedArgs(2, bufTrigW, bufWeights);
  fftW.setFixedArgs(2, bufTrigW);
  fftHin.setFixedArgs(2, bufTrigH);

  fftMidIn.setFixedArgs( 2, bufTrigM);
  fftMidOut.setFixedArgs(2, bufTrigM);
  
  carryB.setFixedArgs(1, bufCarry, bufBitsC);
  tailMulLow.setFixedArgs(3, bufTrigH);
  tailMul.setFixedArgs(3, bufTrigH);
  tailSquareZero.setFixedArgs(2, bufTrigH);
  tailSquare.setFixedArgs(2, bufTrigH);
  kernIsEqual.setFixedArgs(2, bufTrue);

  bufReady.zero();
  bufROE.zero();
  bufStatsCarry.zero();
  bufTrue.write({1});

  if (args.verbose) {
    selftestTrig();
  }

  queue->finish();
}

#if 0
void Gpu::measureTransferSpeed() {
  u32 SIZE_MB = 16;
  vector<double> data(SIZE_MB * 1024 * 1024, 1);
  Buffer<double> buf{profile.make("DMA"), queue, SIZE};

  Timer t;
  for (int i = 0; i < 4; ++i) {
    buf.write(data);
    log("buffer Write : %f GB/s\n", double(SIZE / 1024 / 1024) * sizeof(double) / (1024 * t.reset()));
  }

  for (int i = 0; i < 4; ++i) {
    buf.read(data);
    // queue->finish();
    log("buffer READ : %f GB/s\n", double(SIZE / 1024 / 1024) * sizeof(double) / (1024 * t.reset()));
  }

  queue->finish();
}
#endif

u32 Gpu::updateCarryPos(u32 bit) {
  return (statsBits & bit) && (carryPos < CARRY_SIZE) ? carryPos++ : carryPos;
}

void Gpu::carryFused(Buffer<double>& a, Buffer<double>& b) {
  assert(roePos <= ROE_SIZE);
  roePos < wantROE ? kCarryFusedROE(a, b, roePos++)
                   : kCarryFused(a, b, updateCarryPos(1 << 0));
}

void Gpu::carryFusedMul(Buffer<double>& a, Buffer<double>& b) {
  assert(roePos <= ROE_SIZE);
  roePos < wantROE ? kCarryFusedMulROE(a, b, roePos++)
                   : kCarryFusedMul(a, b, updateCarryPos(1 << 1));
}

void Gpu::carryA(Buffer<int>& a, Buffer<double>& b) {
  assert(roePos <= ROE_SIZE);
  roePos < wantROE ? kCarryAROE(a, b, roePos++)
                   : kCarryA(a, b, updateCarryPos(1 << 2));
}

void Gpu::carryLL(Buffer<int>& a, Buffer<double>& b) { kCarryLL(a, b, updateCarryPos(1 << 2)); }

void Gpu::carryM(Buffer<int>& a, Buffer<double>& b) {
  assert(roePos <= ROE_SIZE);
  roePos < wantROE ? kCarryMROE(a, b, roePos++)
                   : kCarryM(a, b, updateCarryPos(1 << 3));
}

vector<Buffer<i32>> Gpu::makeBufVector(u32 size) {
  vector<Buffer<i32>> r;
  for (u32 i = 0; i < size; ++i) { r.emplace_back(timeBufVect, queue, N); }
  return r;
}

pair<RoeInfo, RoeInfo> Gpu::readROE() {
  assert(roePos <= ROE_SIZE);
  if (roePos) {
    vector<float> roe = bufROE.read(roePos);
    assert(roe.size() == roePos);
    bufROE.zero(roePos);
    roePos = 0;
    auto [squareRoe, mulRoe] = split(roe, mulRoePos);
    mulRoePos.clear();
    return {roeStat(squareRoe), roeStat(mulRoe)};
  } else {
    return {};
  }
}

RoeInfo Gpu::readCarryStats() {
  assert(carryPos <= CARRY_SIZE);
  if (carryPos == 0) { return {}; }
  vector<float> carry = bufStatsCarry.read(carryPos);
  assert(carry.size() == carryPos);
  bufStatsCarry.zero(carryPos);
  carryPos = 0;

  RoeInfo ret = roeStat(carry);

#if 0
  log("%s\n", ret.toString().c_str());

  std::sort(carry.begin(), carry.end());
  File fo = File::openAppend("carry.txt");
  auto it = carry.begin();
  u32 n = carry.size();
  u32 c = 0;
  for (int i=0; i < 500; ++i) {
    double y = 0.23 + (0.48 - 0.23) / 500 * i;
    while (it < carry.end() && *it < y) {
      ++c;
      ++it;
    }
    fo.printf("%f %f\n", y, c / double(n));
  }

  // for (auto x : carry) { fo.printf("%f\n", x); }
  fo.printf("\n\n");
#endif

  return ret;
}

template<typename T>
static bool isAllZero(vector<T> v) { return std::all_of(v.begin(), v.end(), [](T x) { return x == 0;}); }

// Read from GPU, verifying the transfer with a sum, and retry on failure.
vector<int> Gpu::readChecked(Buffer<int>& buf) {
  for (int nRetry = 0; nRetry < 3; ++nRetry) {
    sum64(bufSumOut, u32(buf.size * sizeof(int)), buf);

    vector<u64> expectedVect(1);

    bufSumOut.readAsync(expectedVect);
    vector<int> data = readOut(buf);

    u64 gpuSum = expectedVect[0];

    u64 hostSum = 0;
    for (auto it = data.begin(), end = data.end(); it < end; it += 2) {
      hostSum += u32(*it) | (u64(*(it + 1)) << 32);
    }

    if (hostSum == gpuSum) {
      // A buffer containing all-zero is exceptional, so mark that through the empty vector.
      if (gpuSum == 0 && isAllZero(data)) {
        log("Read ZERO\n");
        return {};
      }
      return data;
    }

    log("GPU read failed: %016" PRIx64 " (gpu) != %016" PRIx64 " (host)\n", gpuSum, hostSum);
  }
  throw "GPU persistent read errors";
}

Words Gpu::readAndCompress(Buffer<int>& buf)  { return compactBits(readChecked(buf), E); }
vector<u32> Gpu::readCheck() { return readAndCompress(bufCheck); }
vector<u32> Gpu::readData() { return readAndCompress(bufData); }

// out := inA * inB;
void Gpu::mul(Buffer<int>& ioA, Buffer<double>& inB, Buffer<double>& tmp1, Buffer<double>& tmp2, bool mul3) {
    fftP(tmp1, ioA);
    fftMidIn(tmp2, tmp1);
    tailMul(tmp1, inB, tmp2);

    // Register the current ROE pos as multiplication (vs. a squaring)
    if (mulRoePos.empty() || mulRoePos.back() < roePos) { mulRoePos.push_back(roePos); }

    fftMidOut(tmp2, tmp1);
    fftW(tmp1, tmp2);
    if (mul3) { carryM(ioA, tmp1); } else { carryA(ioA, tmp1); }
    carryB(ioA);
}

void Gpu::mul(Buffer<int>& io, Buffer<double>& buf1) {
  // We know that mul() stores double output in buf1; so we're going to use buf2 & buf3 for temps.
  mul(io, buf1, buf2, buf3, false);
}

// out := inA * inB;
void Gpu::modMul(Buffer<int>& ioA, Buffer<int>& inB, bool mul3) {
  fftP(buf2, inB);
  fftMidIn(buf1, buf2);
  mul(ioA, buf1, buf2, buf3, mul3);
};

void Gpu::writeState(const vector<u32>& check, u32 blockSize) {
  assert(blockSize > 0);
  writeIn(bufCheck, check);

  bufData << bufCheck;
  bufAux  << bufCheck;
  
  u32 n = 0;
  for (n = 1; blockSize % (2 * n) == 0; n *= 2) {
    squareLoop(bufData, 0, n);
    modMul(bufData, bufAux);
    bufAux << bufData;
  }
  
  assert((n & (n - 1)) == 0);
  assert(blockSize % n == 0);
    
  blockSize /= n;
  assert(blockSize >= 2);
  
  for (u32 i = 0; i < blockSize - 2; ++i) {
    squareLoop(bufData, 0, n);
    modMul(bufData, bufAux);
  }
  
  squareLoop(bufData, 0, n);
  modMul(bufData, bufAux, true);
}
  
bool Gpu::doCheck(u32 blockSize) {
  squareLoop(bufAux, bufCheck, 0, blockSize, true);
  modMul(bufCheck, bufData);
  return isEqual(bufCheck, bufAux);
}

void Gpu::logTimeKernels() {
  auto prof = profile.get();
  u64 total = 0;
  for (const TimeInfo* p : prof) { total += p->times[2]; }
  if (!total) { return; } // no profile
  
  char buf[256];
  // snprintf(buf, sizeof(buf), "Profile:\n ");

  string s = "Profile:\n";
  for (const TimeInfo* p : prof) {
    u32 n = p->n;
    assert(n);
    double f = 1e-3 / n;
    double percent = 100.0 / total * p->times[2];
    if (!args.verbose && percent < 0.2) { break; }
    snprintf(buf, sizeof(buf),
             args.verbose ? "%s %5.2f%% %-11s : %6.0f us/call x %5d calls  (%.3f %.0f)\n"
                          : "%s %5.2f%% %-11s %4.0f x%6d  %.3f %.0f\n",
             logContext().c_str(),
             percent, p->name.c_str(), p->times[2] * f, n, p->times[0] * (f * 1e-3), p->times[1] * (f * 1e-3));
    s += buf;
  }
  log("%s", s.c_str());
  // log("Total time %.3fs\n", total * 1e-9);
  profile.reset();
}

vector<int> Gpu::readOut(Buffer<int> &buf) {
  transpOut(bufAux, buf);
  return bufAux.read();
}

void Gpu::writeIn(Buffer<int>& buf, const vector<u32>& words) { writeIn(buf, expandBits(words, N, E)); }

void Gpu::writeIn(Buffer<int>& buf, vector<i32>&& words) {
  bufAux.write(std::move(words));
  transpIn(buf, bufAux);
}

Words Gpu::expExp2(const Words& A, u32 n) {
  u32 logStep   = 10000;
  u32 blockSize = 100;
  
  writeIn(bufData, std::move(A));
  IterationTimer timer{0};
  u32 k = 0;
  while (k < n) {
    u32 its = std::min(blockSize, n - k);
    squareLoop(bufData, 0, its);
    k += its;
    queue->finish();
    if (k % logStep == 0) {
      float secsPerIt = timer.reset(k);
      log("%u / %u, %.0f us/it\n", k, n, secsPerIt * 1'000'000);
    }
  }
  return readData();
}

// A:= A^h * B
void Gpu::expMul(Buffer<i32>& A, u64 h, Buffer<i32>& B) {
  exponentiate(A, h, buf1, buf2, buf3);
  modMul(A, B);
}

// return A^x * B
Words Gpu::expMul(const Words& A, u64 h, const Words& B, bool doSquareB) {
  writeIn(bufCheck, B);
  if (doSquareB) { square(bufCheck); }

  writeIn(bufData, A);
  expMul(bufData, h, bufCheck);
  return readData();
}

static bool testBit(u64 x, int bit) { return x & (u64(1) << bit); }

void Gpu::bottomHalf(Buffer<double>& out, Buffer<double>& inTmp) {
  fftMidIn(out, inTmp);
  if (!tail_single_kernel) tailSquareZero(inTmp, out);
  tailSquare(inTmp, out);
  fftMidOut(out, inTmp);
}

// See "left-to-right binary exponentiation" on wikipedia
void Gpu::exponentiate(Buffer<int>& bufInOut, u64 exp, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
  if (exp == 0) {
    bufInOut.set(1);
  } else if (exp > 1) {
    fftP(buf3, bufInOut);
    fftMidIn(buf2, buf3);
    fftHin(buf1, buf2); // save "base" to buf1

    int p = 63;
    while (!testBit(exp, p)) { --p; }

    for (--p; ; --p) {
      bottomHalf(buf2, buf3);

      if (testBit(exp, p)) {
        doCarry(buf3, buf2);
        fftMidIn(buf2, buf3);
        tailMulLow(buf3, buf2, buf1);
        fftMidOut(buf2, buf3);
      }

      if (!p) { break; }

      doCarry(buf3, buf2);
    }

    fftW(buf3, buf2);
    carryA(bufInOut, buf3);
    carryB(bufInOut);
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

void Gpu::square(Buffer<int>& out, Buffer<int>& in, bool leadIn, bool leadOut, bool doMul3, bool doLL) {
  // LL does not do Mul3
  assert(!(doMul3 && doLL));

  if (leadIn) { fftP(buf2, in); }
  
  bottomHalf(buf1, buf2);

  if (leadOut) {
    fftW(buf2, buf1);
    if (!doLL && !doMul3) {
      carryA(out, buf2);
    } else if (doLL) {
      carryLL(out, buf2);
    } else {
      carryM(out, buf2);
    }
    carryB(out);
  } else {
    assert(!useLongCarry);
    assert(!doMul3);

    if (doLL) {
      carryFusedLL(buf2, buf1);
    } else {
      carryFused(buf2, buf1);
    }
    // Unused: carryFusedMul(buf2, buf1);
  }
}

void Gpu::square(Buffer<int>& io) { square(io, io, true, true, false, false); }

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

bool Gpu::isEqual(Buffer<int>& in1, Buffer<int>& in2) {
  kernIsEqual(in1, in2);
  int isEq = 0;
  bufTrue.read(&isEq, 1);
  if (!isEq) { bufTrue.write({1}); }
  return isEq;
}
  
u64 Gpu::bufResidue(Buffer<int> &buf) {
  readResidue(bufSmallOut, buf);
  int words[64];
  bufSmallOut.read(words, 64);

  int carry = 0;
  for (int i = 0; i < 32; ++i) { carry = (words[i] + carry < 0) ? -1 : 0; }

  u64 res = 0;
  int hasBits = 0;
  for (int k = 0; k < 32 && hasBits < 64; ++k) {
    u32 len = bitlen(N, E, k);
    int w = words[32 + k] + carry;
    carry = (w < 0) ? -1 : 0;
    if (w < 0) { w += (1 << len); }
    assert(w >= 0 && w < (1 << len));
    res |= u64(w) << hasBits;
    hasBits += len;
  }
  return res;
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

string RoeInfo::toString() const {
  if (!N) { return {}; }

  char buf[256];
  snprintf(buf, sizeof(buf), "Z(%u)=%.1f Max %f mean %f sd %f (%f, %f)",
           N, z(.5f), max, mean, sd, gumbelMiu, gumbelBeta);
  return buf;
}

static string makeLogStr(const string& status, u32 k, u64 res, float secsPerIt, u32 nIters) {
  char buf[256];
  
  snprintf(buf, sizeof(buf), "%2s %9u %016" PRIx64 " %4.0f ETA %s; ",
           status.c_str(), k, res, /* k / float(nIters) * 100, */
           secsPerIt * 1'000'000, getETA(k, nIters, secsPerIt).c_str());
  return buf;
}

void Gpu::doBigLog(u32 k, u64 res, bool checkOK, float secsPerIt, u32 nIters, u32 nErrors) {
  auto [roeSq, roeMul] = readROE();
  double z = roeSq.z();
  zAvg.update(z, roeSq.N);
  log("%sZ=%.0f (avg %.1f)%s\n", makeLogStr(checkOK ? "OK" : "EE", k, res, secsPerIt, nIters).c_str(),
      z, zAvg.avg(), (nErrors ? " "s + to_string(nErrors) + " errors"s : ""s).c_str());

  if (roeSq.N > 2 && z < 20) {
    log("Danger ROE! Z=%.1f is too small, increase precision or FFT size!\n", z);
  }

  // Unless ROE log is not explicitly requested, measure only a few iterations to minimize overhead
  wantROE = args.logROE ? ROE_SIZE : 400;

  RoeInfo carryStats = readCarryStats();
  if (carryStats.N > 2) {
    u32 m = ldexp(carryStats.max, 32);
    double z = carryStats.z();
    log("Carry: %x Z(%u)=%.1f\n", m, carryStats.N, z);
  }
}

bool Gpu::equals9(const Words& a) {
  if (a[0] != 9) { return false; }
  for (auto it = next(a.begin()); it != a.end(); ++it) { if (*it) { return false; }}
  return true;
}

int ulps(double a, double b) {
  if (a == 0 && b == 0) { return 0; }

  u64 aa = as<u64>(a);
  u64 bb = as<u64>(b);
  bool sameSign = (aa >> 63) == (bb >> 63);
  int delta = sameSign ? bb - aa : bb + aa;
  return delta;
}

[[maybe_unused]] static double trigNorm(double c, double s) {
  double c2 = c * c;
  double err = fma(c, c, -c2);
  double norm = c2 + fma(s, s, err);
  return norm;
}

void Gpu::selftestTrig() {
  const u32 n = hN / 8;
  testTrig(buf1);
  vector<double> trig = buf1.read(n * 2);
  int sup = 0, sdown = 0;
  int cup = 0, cdown = 0;
  int oneUp = 0, oneDown = 0;
  for (u32 k = 0; k < n; ++k) {
    double c = trig[2*k];
    double s = trig[2*k + 1];

#if 0
    auto [refCos, refSin] = root1(hN, k);
#else
    long double angle = M_PIl * k / (hN/2);
    double refSin = sinl(angle);
    double refCos = cosl(angle);
#endif

    if (s > refSin) { ++sup; }
    if (s < refSin) { ++sdown; }
    if (c > refCos) { ++cup; }
    if (c < refCos) { ++cdown; }
    
    double norm = trigNorm(c, s);
    
    if (norm < 1.0) { ++oneDown; }
    if (norm > 1.0) { ++oneUp; }
  }

  log("TRIG sin(): imperfect %d / %d (%.2f%%), balance %d\n",
      sup + sdown, n, (sup + sdown) * 100.0 / n, sup - sdown);
  log("TRIG cos(): imperfect %d / %d (%.2f%%), balance %d\n",
      cup + cdown, n, (cup + cdown) * 100.0 / n, cup - cdown);
  log("TRIG norm: up %d, down %d\n", oneUp, oneDown);
  
  if (isAmdGpu(queue->context->deviceId())) {
    vector<string> WHATS {"V_NOP", "V_ADD_I32", "V_FMA_F32", "V_ADD_F64", "V_FMA_F64", "V_MUL_F64", "V_MAD_U64_U32"};
    for (int w = 0; w < int(WHATS.size()); ++w) {
      const int what = w;
      testTime(what, bufCarry);
      vector<i64> times = bufCarry.read(4096 * 2);
      [[maybe_unused]] i64 prev = 0;
      u64 min = -1;
      u64 sum = 0;
      for (int i = 0; i < int(times.size()); ++i) {
        i64 x = times[i];
#if 0
        if (x != prev) {
          log("%4d : %ld\n", i, x);
          prev = x;
        }
#endif
        if (x > 0 && u64(x) < min) { min = x; }
        if (x > 0) { sum += x; }
      }
      log("%-15s : %.2f cycles latency; time min: %d; avg %.0f\n",
          WHATS[w].c_str(), double(min - 40) / 48, int(min), double(sum) / times.size());
    }
  }
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

fs::path Gpu::saveProof(const Args& args, const ProofSet& proofSet) {
  for (int retry = 0; retry < 2; ++retry) {
    auto [proof, hashes] = proofSet.computeProof(this);
    fs::path tmpFile = proof.file(args.proofToVerifyDir);
    proof.save(tmpFile);
            
    fs::path proofFile = proof.file(args.proofResultDir);

    bool ok = Proof::load(tmpFile).verify(this, hashes);
    log("Proof '%s' verification %s\n", tmpFile.string().c_str(), ok ? "OK" : "FAILED");
    if (ok) {
      fancyRename(tmpFile, proofFile);
      log("Proof '%s' generated\n", proofFile.string().c_str());
      return proofFile;
    }
  }
  throw "bad proof generation";
}

PRPState Gpu::loadPRP(Saver<PRPState>& saver) {
  for (int nTries = 0; nTries < 2; ++nTries) {
    if (nTries) {
      saver.dropMostRecent();    // Try an earlier savefile
    }

    PRPState state = saver.load();
    writeState(state.check, state.blockSize);
    u64 res = dataResidue();

    if (res == state.res64) {
      log("OK %9u on-load: blockSize %d, %016" PRIx64 "\n", state.k, state.blockSize, res);
      return state;
      // return {loaded.k, loaded.blockSize, loaded.nErrors};
    }

    log("EE %9u on-load: %016" PRIx64 " vs. %016" PRIx64 "\n", state.k, res, state.res64);

    if (!state.k) { break; }  // We failed on PRP start
  }

  throw "Error on load";
}

u32 Gpu::getProofPower(u32 k) {
  u32 power = ProofSet::effectivePower(E, args.getProofPow(E), k);

  if (power != args.getProofPow(E)) {
    log("Proof using power %u (vs %u)\n", power, args.getProofPow(E));
  }

  if (!power) {
    log("Proof generation disabled!\n");
  } else {
    log("Proof of power %u requires about %.1fGB of disk space\n", power, ProofSet::diskUsageGB(E, power));
  }
  return power;
}

tuple<bool, RoeInfo> Gpu::measureCarry() {
  u32 blockSize{}, iters{}, warmup{};

  blockSize = 200;
  iters = 2000;
  warmup = 50;

  assert(iters % blockSize == 0);

  u32 k = 0;
  PRPState state{E, 0, blockSize, 3, makeWords(E, 1), 0};
  writeState(state.check, state.blockSize);
  {
    u64 res = dataResidue();
    if (res != state.res64) {
      log("residue expected %016" PRIx64 " found %016" PRIx64 "\n", state.res64, res);
    }
    assert(res == state.res64);
  }

  modMul(bufCheck, bufData);
  square(bufData, bufData, true, useLongCarry);
  ++k;

  while (k < warmup) {
    square(bufData, bufData, useLongCarry, useLongCarry);
    ++k;
  }

  readCarryStats(); // ignore the warm-up iterations

  if (Signal::stopRequested()) { throw "stop requested"; }

  bool leadIn = useLongCarry;
  while (true) {
    while (k % blockSize < blockSize-1) {
      square(bufData, bufData, leadIn, useLongCarry);
      ++k;
      leadIn = useLongCarry;
    }
    square(bufData, bufData, useLongCarry, true);
    leadIn = true;
    ++k;

    if (k >= iters) { break; }

    modMul(bufCheck, bufData);
    if (Signal::stopRequested()) { throw "stop requested"; }
  }

  [[maybe_unused]] u64 res = dataResidue();
  if (Signal::stopRequested()) { throw "stop requested"; }

  bool ok = doCheck(blockSize);
  auto stats = readCarryStats();

  // log("%s %016" PRIx64 " %s\n", ok ? "OK" : "EE", res, roe.toString(statsBits).c_str());
  return {ok, stats};
}

tuple<bool, u64, RoeInfo, RoeInfo> Gpu::measureROE(bool quick) {
  u32 blockSize{}, iters{}, warmup{};

  if (true) {
    blockSize = 200;
    iters = 2000;
    warmup = 50;
  } else {
    blockSize = 500;
    iters = 10'000;
    warmup = 100;
  }

  assert(iters % blockSize == 0);

  wantROE = ROE_SIZE; // should be large enough to capture fully this measureROE()

  u32 k = 0;
  PRPState state{E, 0, blockSize, 3, makeWords(E, 1), 0};
  writeState(state.check, state.blockSize);
  {
    u64 res = dataResidue();
    if (res != state.res64) {
      log("residue expected %016" PRIx64 " found %016" PRIx64 "\n", state.res64, res);
    }
    assert(res == state.res64);
  }

  modMul(bufCheck, bufData);
  square(bufData, bufData, true, useLongCarry);
  ++k;

  while (k < warmup) {
    square(bufData, bufData, useLongCarry, useLongCarry);
    ++k;
  }

  readROE(); // ignore the warm-up iterations

  if (Signal::stopRequested()) { throw "stop requested"; }

  bool leadIn = useLongCarry;
  while (true) {
    while (k % blockSize < blockSize-1) {
      square(bufData, bufData, leadIn, useLongCarry);
      ++k;
      leadIn = useLongCarry;
    }
    square(bufData, bufData, useLongCarry, true);
    leadIn = true;
    ++k;

    if (k >= iters) { break; }

    modMul(bufCheck, bufData);
    if (Signal::stopRequested()) { throw "stop requested"; }
  }

  [[maybe_unused]] u64 res = dataResidue();
  if (Signal::stopRequested()) { throw "stop requested"; }

  bool ok = doCheck(blockSize);
  auto roes = readROE();

  wantROE = 0;
  // log("%s %016" PRIx64 " %s\n", ok ? "OK" : "EE", res, roe.toString(statsBits).c_str());
  return {ok, res, roes.first, roes.second};
}

double Gpu::timePRP() {
  u32 blockSize{}, iters{}, warmup{};

  if (true) {
    blockSize = 200;
    iters = 1000;
    warmup = 30;
  } else {
    blockSize = 1000;
    iters = 10'000;
    warmup = 100;
  }

  assert(iters % blockSize == 0);

  u32 k = 0;
  PRPState state{E, 0, blockSize, 3, makeWords(E, 1), 0};
  writeState(state.check, state.blockSize);
  assert(dataResidue() == state.res64);

  modMul(bufCheck, bufData);
  square(bufData, bufData, true, useLongCarry);
  ++k;

  while (k < warmup) {
    square(bufData, bufData, useLongCarry, useLongCarry);
    ++k;
  }
  queue->finish();
  if (Signal::stopRequested()) { throw "stop requested"; }

  Timer t;
  bool leadIn = useLongCarry;
  while (true) {
    while (k % blockSize < blockSize-1) {
      square(bufData, bufData, leadIn, useLongCarry);
      ++k;
      leadIn = useLongCarry;
    }
    square(bufData, bufData, useLongCarry, true);
    leadIn = true;
    ++k;

    if (k >= iters) { break; }

    modMul(bufCheck, bufData);
    if (Signal::stopRequested()) { throw "stop requested"; }
  }
  queue->finish();
  double secsPerIt = t.reset() / (iters - warmup);

  if (Signal::stopRequested()) { throw "stop requested"; }

  u64 res = dataResidue();
  bool ok = doCheck(blockSize);
  if (!ok) {
    log("Error %016" PRIx64 "\n", res);
    secsPerIt = 0.1; // a large value to mark the error
  }
  return secsPerIt * 1e6;
}

PRPResult Gpu::isPrimePRP(const Task& task) {
  const constexpr u32 LOG_STEP = 20'000; // log every 20k its
  assert(E == task.exponent);

  // This timer is used to measure total elapsed time to be written to the savefile.
  Timer elapsedTimer;

  u32 nErrors = 0;
  int nSeqErrors = 0;
  u64 lastFailedRes64 = 0;
  
 reload:
  elapsedTimer.reset();
  u32 blockSize{}, k{};
  double elapsedBefore = 0;

  {
    PRPState state = loadPRP(*getSaver());
    nErrors = std::max(nErrors, state.nErrors);
    blockSize = state.blockSize;
    k = state.k;
    elapsedBefore = state.elapsed;
  }

  assert(blockSize > 0 && LOG_STEP % blockSize == 0);

  u32 checkStep = checkStepForErrors(blockSize, nErrors);
  assert(checkStep % LOG_STEP == 0);

  u32 power = getProofPower(k);
  
  ProofSet proofSet{E, power};

  bool isPrime = false;

  u64 finalRes64 = 0;
  vector<u32> res2048;

  // We extract the res64 at kEnd.
  // For M=2^E-1, residue "type-3" == 3^(M+1), and residue "type-1" == type-3 / 9,
  // See http://www.mersenneforum.org/showpost.php?p=468378&postcount=209
  // For both type-1 and type-3 we need to do E squarings (as M+1==2^E).
  const u32 kEnd = E;
  assert(k < kEnd);

  // We continue beyound kEnd: to the next multiple of blockSize, to do a check there
  u32 kEndEnd = roundUp(kEnd, blockSize);

  bool skipNextCheckUpdate = false;

  u32 persistK = proofSet.next(k);
  bool leadIn = true;

  assert(k % blockSize == 0);
  assert(checkStep % blockSize == 0);

  const u32 startK = k;
  IterationTimer iterationTimer{k};

  wantROE = 0; // skip the initial iterations

  while (true) {
    assert(k < kEndEnd);
    
    if (!wantROE && k - startK > 30) { wantROE = args.logROE ? ROE_SIZE : 2'000; }

    if (skipNextCheckUpdate) {
      skipNextCheckUpdate = false;
    } else if (k % blockSize == 0) {
      assert(leadIn);
      modMul(bufCheck, bufData);
    }

    ++k; // !! early inc

    bool doStop = (k % blockSize == 0) && (Signal::stopRequested() || (args.iters && k - startK >= args.iters));
    bool leadOut = (k % blockSize == 0) || k == persistK || k == kEnd || useLongCarry;

    assert(!doStop || leadOut);
    if (doStop) { log("Stopping, please wait..\n"); }

    square(bufData, bufData, leadIn, leadOut, false);
    leadIn = leadOut;
    
    if (k == persistK) {
      vector<int> rawData = readChecked(bufData);
      if (rawData.empty()) {
        log("Data error ZERO\n");
        ++nErrors;
        goto reload;
      }
      (*background)([=, E=this->E] { ProofSet::save(E, power, k, compactBits(rawData, E)); });
      persistK = proofSet.next(k);
    }

    if (k == kEnd) {
      Words words = readData();
      isPrime = equals9(words);
      doDiv9(E, words);
      finalRes64 = residue(words);
      res2048.clear();
      assert(words.size() >= 64);
      res2048.insert(res2048.end(), words.begin(), std::next(words.begin(), 64));
      log("%s %8d / %d, %s\n", isPrime ? "PP" : "CC", kEnd, E, hex(finalRes64).c_str());
    }

    bool doCheck = doStop || (k % checkStep == 0) || (k >= kEndEnd) || (k - startK == 2 * blockSize);
    bool doLog = k % LOG_STEP == 0;

    if (!leadOut || (!doCheck && !doLog)) {
      if (k % args.flushStep == 0) { queue->finish(); }
      continue;
    }

    assert(doCheck || doLog);

    u64 res = dataResidue();
    float secsPerIt = iterationTimer.reset(k);

    vector<int> rawCheck = readChecked(bufCheck);
    if (rawCheck.empty()) {
      ++nErrors;
      log("%9u %016" PRIx64 " read NULL check\n", k, res);
      if (++nSeqErrors > 2) { throw "sequential errors"; }
      goto reload;
    }

    if (!doCheck) {
      (*background)([=, this] {
        getSaver()->saveUnverified({E, k, blockSize, res, compactBits(rawCheck, E), nErrors,
                                    elapsedBefore + elapsedTimer.at()});
      });

      log("   %9u %016" PRIx64 " %4.0f\n", k, res, /*k / float(kEndEnd) * 100*,*/ secsPerIt * 1'000'000);
      RoeInfo carryStats = readCarryStats();
      if (carryStats.N) {
        u32 m = ldexp(carryStats.max, 32);
        double z = carryStats.z();
        log("Carry: %x Z(%u)=%.1f\n", m, carryStats.N, z);
      }
    } else {
      bool ok = this->doCheck(blockSize);
      [[maybe_unused]] float secsCheck = iterationTimer.reset(k);

      if (ok) {
        nSeqErrors = 0;
        // lastFailedRes64 = 0;
        skipNextCheckUpdate = true;

        if (k < kEnd) {
          (*background)([=, this, rawCheck = std::move(rawCheck)] {
            getSaver()->save({E, k, blockSize, res, compactBits(rawCheck, E), nErrors, elapsedBefore + elapsedTimer.at()});
          });
        }

        doBigLog(k, res, ok, secsPerIt, kEndEnd, nErrors);
          
        if (k >= kEndEnd) {
          fs::path proofFile = saveProof(args, proofSet);
          return {isPrime, finalRes64, nErrors, proofFile.string(), toHex(res2048)};
        }        
      } else {
        ++nErrors;
        doBigLog(k, res, ok, secsPerIt, kEndEnd, nErrors);
        if (++nSeqErrors > 2) {
          log("%d sequential errors, will stop.\n", nSeqErrors);
          throw "too many errors";
        }
        if (res == lastFailedRes64) {
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

LLResult Gpu::isPrimeLL(const Task& task) {
  assert(E == task.exponent);
  wantROE = 0;

  Timer elapsedTimer;

  Saver<LLState> saver{E, 1000, args.nSavefiles};

  reload:
  elapsedTimer.reset();

  u32 startK = 0;
  double elapsedBefore = 0;
  {
    LLState state = saver.load();

    elapsedBefore = state.elapsed;
    startK = state.k;
    u64 expectedRes = (u64(state.data[1]) << 32) | state.data[0];
    writeIn(bufData, std::move(state.data));
    u64 res = dataResidue();
    if (res != expectedRes) { throw "Invalid savefile (res64)"; }
    assert(res == expectedRes);
    log("LL loaded @ %u : %016" PRIx64 "\n", startK, res);
  }

  IterationTimer iterationTimer{startK};

  u32 k = startK;
  u32 kEnd = E - 2;
  bool leadIn = true;

  while (true) {
    ++k;
    bool doStop = (k >= kEnd) || (args.iters && k - startK >= args.iters);

    if (Signal::stopRequested()) {
      doStop = true;
      log("Stopping, please wait..\n");
    }

    bool doLog = (k % 10'000 == 0) || doStop;
    bool leadOut = doLog || useLongCarry;

    squareLL(bufData, leadIn, leadOut);
    leadIn = leadOut;

    if (!doLog) {
      if (k % args.flushStep == 0) { queue->finish(); } // Periodically flush the queue
      continue;
    }

    u64 res64 = 0;
    auto data = readData();
    bool isAllZero = data.empty();

    if (isAllZero) {
      if (k < kEnd) {
        log("Error: early ZERO @ %u\n", k);
        if (doStop) {
          throw "stop requested";
        } else {
          goto reload;
        }
      }
      res64 = 0;
    } else {
      assert(data.size() >= 2);
      res64 = (u64(data[1]) << 32) | data[0];
      saver.save({E, k, std::move(data), elapsedBefore + elapsedTimer.at()});
    }

    float secsPerIt = iterationTimer.reset(k);
    log("%9u %016" PRIx64 " %4.0f\n", k, res64, secsPerIt * 1'000'000);

    if (k >= kEnd) { return {isAllZero, res64}; }

    if (doStop) { throw "stop requested"; }
  }
}

array<u64, 4> Gpu::isCERT(const Task& task) {
  assert(E == task.exponent);
  wantROE = 0;

  // Get CERT start value
  char fname[32];
  sprintf(fname, "M%u.cert", E);
  File fi = File::openReadThrow(fname);

//We need to gracefully handle the CERT file missing.  There is a window in primenet.py between worktodo.txt entry and starting value download.

  u32 nBytes = (E - 1) / 8 + 1;
  Words B = fi.readBytesLE(nBytes);

  writeIn(bufData, std::move(B));

  Timer elapsedTimer;

  elapsedTimer.reset();

  u32 startK = 0;

  IterationTimer iterationTimer{startK};

  u32 k = 0;
  u32 kEnd = task.squarings;
  bool leadIn = true;

  while (true) {
    ++k;
    bool doStop = (k >= kEnd);

    if (Signal::stopRequested()) {
      doStop = true;
      log("Stopping, please wait..\n");
    }

    bool doLog = (k % 100'000 == 0) || doStop;
    bool leadOut = doLog || useLongCarry;

    squareCERT(bufData, leadIn, leadOut);
    leadIn = leadOut;

    if (!doLog) {
      if (k % args.flushStep == 0) { queue->finish(); } // Periodically flush the queue
      continue;
    }

    Words data = readData();
    assert(data.size() >= 2);
    u64 res64 = (u64(data[1]) << 32) | data[0];

    float secsPerIt = iterationTimer.reset(k);
    log("%9u %016" PRIx64 " %4.0f\n", k, res64, secsPerIt * 1'000'000);

    if (k >= kEnd) {
      fs::remove (fname);
      return std::move(SHA3{}.update(data.data(), (E-1)/8+1)).finish();
    }

    if (doStop) { throw "stop requested"; }
  }
}


void Gpu::clear(bool isPRP) {
  if (isPRP) {
    Saver<PRPState>::clear(E);
  } else {
    Saver<LLState>::clear(E);
  }
}

Saver<PRPState> *Gpu::getSaver() {
  if (!saver) { saver = make_unique<Saver<PRPState>>(E, args.blockSize, args.nSavefiles); }
  return saver.get();
}

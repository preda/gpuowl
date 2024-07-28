// Copyright Mihai Preda and George Woltman.

#include "Gpu.h"
#include "Proof.h"
#include "TimeInfo.h"
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

#include <algorithm>
#include <bitset>
#include <limits>
#include <iomanip>
#include <array>
#include <cinttypes>

namespace {

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
  vector<double> weightsIF;
  for (u32 thread = 0; thread < groupWidth; ++thread) {
    auto iw = invWeight(N, E, H, 0, thread, 0);
    // weightsIF.push_back(iw - 1);
    weightsIF.push_back(2 * boundUnderOne(iw));
    auto w = weight(N, E, H, 0, thread, 0);
    // weightsIF.push_back(w - 1);
    weightsIF.push_back(2 * w);
  }

  // the group order matches CarryA/M (not fftP/CarryFused).
  // vector<double> carryWeightsIF;
  for (u32 gy = 0; gy < H; ++gy) {
    auto iw = invWeight(N, E, H, gy, 0, 0);
    weightsIF.push_back(iw - 1);
    
    auto w = weight(N, E, H, gy, 0, 0);
    weightsIF.push_back(w - 1);
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

  return Weights{weightsIF, bits, bitsC};
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

string clDefines(const Args& args, cl_device_id id, FFTConfig fft, const vector<KeyVal>& extraConf, u32 E, bool doLog) {
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
                            });
    if (!isValid) {
      log("Unrecognized -use key '%s'\n", k.c_str());
    }
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

  defines += toDefine("WEIGHT_STEP", double(weight(N, E, fft.shape.height * fft.shape.middle, 0, 0, 1) - 1));
  defines += toDefine("IWEIGHT_STEP", double(invWeight(N, E, fft.shape.height * fft.shape.middle, 0, 0, 1) - 1));
  defines += toDefine("FFT_VARIANT", fft.variant);

  return defines;
}

} // namespace

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
  compiler{args, queue->context, clDefines(args, queue->context->deviceId(), fft, extraConf, E, logFftSize)},
  
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

  K(tailSquare,    "tailsquare.cl", "tailSquare", hN / nH / 2),
  K(tailMul,       "tailmul.cl", "tailMul",       hN / nH / 2),
  K(tailMulLow,    "tailmul.cl", "tailMul",       hN / nH / 2, "-DMUL_LOW=1"),
  
  // 256
  K(fftMidIn,  "fftmiddlein.cl",  "fftMiddleIn",  hN / (BIG_H / SMALL_H)),
  K(fftMidOut, "fftmiddleout.cl", "fftMiddleOut", hN / (BIG_H / SMALL_H)),
  
  // 64
  K(transpIn,  "transpose.cl", "transposeIn",  hN / 64),
  K(transpOut, "transpose.cl", "transposeOut", hN / 64),
  
  K(readResidue, "etc.cl", "readResidue", 64, "-DREADRESIDUE=1"),

  // 256
  K(kernIsEqual, "etc.cl", "isEqual", 256 * 256, "-DISEQUAL=1"),
  K(sum64,       "etc.cl", "sum64",   256 * 256, "-DSUM64=1"),
#undef K

  bufTrigW{shared.bufCache->smallTrig(WIDTH, nW)},
  bufTrigH{shared.bufCache->smallTrig(SMALL_H, nH)},
  bufTrigM{shared.bufCache->middleTrig(SMALL_H, BIG_H / SMALL_H, WIDTH)},

  bufTrigBHW{shared.bufCache->trigBHW(WIDTH, hN, BIG_H)},
  bufTrigSquare(shared.bufCache->trigSquare(hN, nH, SMALL_H)),

  weights{genWeights(E, WIDTH, BIG_H, nW)},

  bufWeights{q->context, std::move(weights.weightsIF)},
  bufBits{q->context,    std::move(weights.bitsCF)},
  bufBitsC{q->context,   std::move(weights.bitsC)},

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

  BUF(buf1, N),
  BUF(buf2, N),
  BUF(buf3, N),
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

  useLongCarry = useLongCarry || (bitsPerWord < 10.5f);

  if (useLongCarry) { log("Using long carry!\n"); }
  
  for (Kernel* k : {&kCarryFused, &kCarryFusedROE, &kCarryFusedMul, &kCarryFusedMulROE, &kCarryFusedLL}) {
    k->setFixedArgs(3, bufCarry, bufReady, bufTrigW, bufBits, bufWeights);
  }

  for (Kernel* k : {&kCarryFusedROE, &kCarryFusedMulROE})           { k->setFixedArgs(8, bufROE); }
  for (Kernel* k : {&kCarryFused, &kCarryFusedMul, &kCarryFusedLL}) { k->setFixedArgs(8, bufStatsCarry); }

  for (Kernel* k : {&kCarryA, &kCarryAROE, &kCarryM, &kCarryMROE, &kCarryLL}) {
    k->setFixedArgs(3, bufCarry, bufBitsC, bufWeights);
  }

  for (Kernel* k : {&kCarryAROE, &kCarryMROE})      { k->setFixedArgs(6, bufROE); }
  for (Kernel* k : {&kCarryA, &kCarryM, &kCarryLL}) { k->setFixedArgs(6, bufStatsCarry); }

  fftP.setFixedArgs(2, bufTrigW, bufWeights);
  fftW.setFixedArgs(2, bufTrigW);
  fftHin.setFixedArgs(2, bufTrigH);

  fftMidIn.setFixedArgs( 2, bufTrigM, bufTrigBHW);
  fftMidOut.setFixedArgs(2, bufTrigM, bufTrigBHW);
  
  carryB.setFixedArgs(1, bufCarry, bufBitsC);
  tailMulLow.setFixedArgs(3, bufTrigH, bufTrigSquare);
  tailMul.setFixedArgs(3, bufTrigH, bufTrigSquare);
  tailSquare.setFixedArgs(2, bufTrigH, bufTrigSquare);
  kernIsEqual.setFixedArgs(2, bufTrue);

  bufReady.zero();
  bufROE.zero();
  bufStatsCarry.zero();
  bufTrue.write({1});
  queue->finish();
}

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

vector<int> Gpu::readSmall(Buffer<int>& buf, u32 start) {
  readResidue(bufSmallOut, buf, start);
  return bufSmallOut.read(128);
}

namespace {

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

} // namespace

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
  bool isEq = bufTrue.read(1)[0];
  if (!isEq) { bufTrue.write({1}); }
  return isEq;
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

  wantROE = 400; // only a few iterations for low overhead

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

}

// ----

fs::path Gpu::saveProof(const Args& args, const ProofSet& proofSet) {
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

std::tuple<u32, u32, u32> Gpu::loadPRP(Saver<PRPState>& saver) {
  // We do at most 4 attempts: the most recent savefile twice, plus two more.
  for (int nTries = 0; nTries < 4; ++nTries) {
    PRPState loaded = saver.load();
    writeState(loaded.check, loaded.blockSize);
    u64 res = dataResidue();

    if (res == loaded.res64) {
      log("OK %9u on-load: blockSize %d, %016" PRIx64 "\n", loaded.k, loaded.blockSize, res);
      return {loaded.k, loaded.blockSize, loaded.nErrors};
    }

    log("EE %9u on-load: %016" PRIx64 " vs. %016" PRIx64 "\n", loaded.k, res, loaded.res64);

    // Attempt to load the first savefile twice (to verify res64 reproducibility)
    if (!nTries) { continue; }

    if (!loaded.k) { break; }  // We failed on PRP start

    saver.dropMostRecent();    // Try an earlier savefile
  }

  throw "Error on load";
}

u32 Gpu::getProofPower(u32 k) {
  u32 power = ProofSet::effectivePower(E, args.getProofPow(E), k);

  if (power != args.getProofPow(E)) {
    log("Proof using power %u (vs %u)\n", power, args.getProofPow(E));
  }

  if (!power) {
    log("Proof generation disabled\n");
  } else {
    if (power > ProofSet::bestPower(E)) {
      log("Warning: proof power %u is excessively large; use at most power %u\n", power, ProofSet::bestPower(E));
    }

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

  u32 nErrors = 0;
  int nSeqErrors = 0;
  
 reload:
  auto [k, blockSize, nLoadErrors] = loadPRP(*getSaver());
  nErrors = std::max(nErrors, nLoadErrors);

  assert(blockSize > 0 && LOG_STEP % blockSize == 0);
  
  u32 checkStep = checkStepForErrors(blockSize, nErrors);
  assert(checkStep % LOG_STEP == 0);

  u32 power = getProofPower(k);
  
  ProofSet proofSet{E, power};

  bool isPrime = false;

  u64 finalRes64 = 0;

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

  u64 lastFailedRes64 = 0;

  while (true) {
    assert(k < kEndEnd);
    
    if (!wantROE && k - startK > 30) { wantROE = 2'000; }

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
        getSaver()->saveUnverified({E, k, blockSize, res, compactBits(rawCheck, E), nErrors});
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
            getSaver()->save({E, k, blockSize, res, compactBits(rawCheck, E), nErrors});
          });
        }

        doBigLog(k, res, ok, secsPerIt, kEndEnd, nErrors);
          
        if (k >= kEndEnd) {
          fs::path proofFile = saveProof(args, proofSet);
          return {isPrime, finalRes64, nErrors, proofFile.string()};
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

  Saver<LLState> saver{E, 0};
  reload:

  u32 startK = 0;
  {
    LLState state = saver.load();

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
      saver.save({E, k, std::move(data)});
    }

    float secsPerIt = iterationTimer.reset(k);
    log("%9u %016" PRIx64 " %4.0f\n", k, res64, secsPerIt * 1'000'000);

    if (k >= kEnd) { return {isAllZero, res64}; }

    if (doStop) { throw "stop requested"; }
  }
}

Saver<PRPState> *Gpu::getSaver() {
  if (!saver) { saver = make_unique<Saver<PRPState>>(E, args.blockSize); }
  return saver.get();
}

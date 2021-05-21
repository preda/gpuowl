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

extern const char *CL_SOURCE;

struct Weights {
  vector<u64> dWeights;
  vector<u64> iWeights;
};

const constexpr u64 PRIME = 0xffffffff00000001u;

// Below: roots of order 2^26 of 1, 2 and their inverses.
// These were computed in pari-gp with sqrtn()
const constexpr u64 HALF      = 0x7fffffff'80000001u; //  9'223'372'034'707'292'161
const constexpr u64 ROOT1     = 0xed41d05b'78d6e286u; // 17'096'174'751'763'063'430
const constexpr u64 ROOT2     = 0x4b5b85a7'c6560773u; // 05'430'080'731'358'824'307
const constexpr u64 INV_ROOT1 = 0xea52f593'bb20759au; // 16'884'827'967'813'875'098
const constexpr u64 INV_ROOT2 = 0x0914970f'f8d65ef5u; //    654'313'940'730'666'741

u64 reduce64(u64 a) { return (a >= PRIME) ? a - PRIME : a; }

u64 neg(u64 x) { return x ? PRIME - x : x; }
u64 mul64(u32 x) { return u64(x) * 0xffffffffu; }
u64 mul96(u32 x) { return neg(x); }

u64 add(u64 a, u64 b) {
  u64 s = a + b;
  return reduce64(s) + ((s >= a) ? 0 : 0xffffffffu);
}

u64 reduce128(u128 x) { return add(add(u64(x), mul64(x >> 64)), mul96(x >> 96)); }

u64 mul(u64 a, u64 b) { return reduce128(u128(a) * b); }
u64 sq(u64 a) { return mul(a, a); }
u64 mul1T4(u64 x) { return reduce128(u128(x) << 48); }
u64 mul3T4(u64 x) { return mul(x, 0xfffeffff00000001u); }

u64 exp(u64 x, u32 e) {
  if (!e) { return 1; }
  u64 r = x;
  u32 nZeros = __builtin_clz(e);
  assert(nZeros < 32);
  u32 nSkip = nZeros + 1;
  e <<= nSkip;
  u32 n = 32 - nSkip;
  for (u32 i = 0; i < n; ++i) {
    r = sq(r);
    if (e >> 31) { r = mul(r, x); }
    e <<= 1;
  }
  return r;
}

namespace {

template<u64 ROOT, u64 INV_ROOT>
u64 root(u32 N, i32 k) {
  assert((N & (N-1)) == 0); // power of 2.

  u32 log2N = __builtin_ctz(N);
  assert(log2N && log2N <= 26);

  return exp((k >= 0) ? ROOT : INV_ROOT, (1 << (26 - log2N)) * abs(k));
}


// Returns the primitive root of unity of order N, to the power k.
u64 root1(u32 N, i32 k) { return root<ROOT1, INV_ROOT1>(N, k); }
u64 root2(u32 N, i32 k) { return root<ROOT2, INV_ROOT2>(N, k); }

/*
u64* smallTrigBlock(u32 W, u32 H, u64* p) {
  for (u32 line = 1; line < H; ++line) {
    for (u32 col = 0; col < W; ++col) {
      *p++ = root1(W * H, line * col);
    }
  }
  return p;
}
*/

ConstBuffer<u64> genSmallTrig(const Context& context, u32 size, u32 radix, bool inverse) {
  vector<u64> tab;
  for (u32 line = 1; line < radix; ++line) {
    for (u32 col = 0; col < size / radix; ++col) {
      tab.push_back(root1(size, i32(col * line) * (inverse ? -1 : 1)));
    }
  }
  // tab.resize(size);
  return {context, "smallTrig", tab};
}

ConstBuffer<u64> genBigTrig(const Context& context, u32 width, u32 height, bool inverse) {
  vector<u64> tab;
  u32 N = width * height;
  for (u32 gr = 0; gr < height / 4; ++gr) {
    for (u32 line = 0; line < 64; ++line) {
      for (u32 col = 0; col < 4; ++col) {
        i32 a = line * (gr * 4 + col);
        tab.push_back(root1(N, a * (inverse ? -1 : 1)));
      }
    }
  }
  return {context, "bigTrig", tab};
}

ConstBuffer<u64> genBigTrigStep(const Context& context, u32 width, u32 height, bool inverse) {
  vector<u64> tab;
  u32 N = width * height;
  for (u32 gr = 0; gr < height / 4; ++gr) {
    for (u32 col = 0; col < 4; ++col) {
      i32 a = 64 * (gr * 4 + col);
      tab.push_back(root1(N, a * (inverse ? -1 : 1)));
    }
  }
  return {context, "bigTrigStep", tab};
}

u32 kAt(u32 H, u32 line, u32 col) { return line + col * H; }

u64 weight(u32 N, u32 E, u32 H, u32 line, u32 col) { return root2(N, extra(N, E, kAt(H, line, col))); }

u64 invWeight(u32 N, u32 E, u32 H, u32 line, u32 col) { return root2(N, -extra(N, E, kAt(H, line, col))); }

Weights genWeights(u32 E, u32 W, u32 H) {
  u32 N = W * H;

  vector<u64> iWeights, dWeights;
  
  for (u32 line = 0; line < H; line += 4) {
    for (u32 col = 0; col < W; ++col) {
      iWeights.push_back(invWeight(N, E, H, line, col));
      dWeights.push_back(weight(N, E, H, line, col));
    }
  }
  return {dWeights, iWeights};
}

string toLiteral(u32 value) { return to_string(value) + 'u'; }
string toLiteral(i32 value) { return to_string(value); }
string toLiteral(u64 value) {
  // return to_string(value) + "ul";
  std::ostringstream ss;
  ss << "0x" << std::hex << std::setprecision(16) << value << "ul";
  return std::move(ss).str();
}

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

cl_program compile(const Args& args, cl_context context, cl_device_id id, u32 N, u32 E, u32 WIDTH) {
  u32 HEIGHT = N / WIDTH;
  
  string clArgs = args.dump.empty() ? ""s : (" -save-temps="s + args.dump + "/" + numberK(N));
  if (!args.safeMath) { clArgs += " -cl-unsafe-math-optimizations"; }
  
  vector<Define> defines =
    {{"EXP", E},
     {"WIDTH", WIDTH},
     {"HEIGHT", HEIGHT}
    };

  if (isAmdGpu(id)) { defines.push_back({"AMDGPU", 1}); }

  defines.push_back({"DWSTEP", weight(N, E, HEIGHT, 1, 0)});
  defines.push_back({"DWSTEP_2", mul(weight(N, E, HEIGHT, 1, 0), HALF)});
  
  defines.push_back({"IWSTEP", invWeight(N, E, HEIGHT, 1, 0)});
  defines.push_back({"IWSTEP_2", mul(invWeight(N, E, HEIGHT, 1, 0), HALF)});
    
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

Gpu::Gpu(const Args& args, u32 E, u32 W, u32 HEIGHT, u32 nW, u32 nH, cl_device_id device, bool timeKernels)
  : Gpu{args, E, W, HEIGHT, nW, nH, device, timeKernels, genWeights(E, W, HEIGHT)}
{}

using float2 = pair<float, float>;

Gpu::Gpu(const Args& args, u32 E, u32 WIDTH, u32 HEIGHT, u32 nW, u32 nH,
         cl_device_id device, bool timeKernels, Weights&& weights) :
  E(E),
  N(WIDTH * HEIGHT),
  nW(nW),
  nH(nH),
  bufSize(N * sizeof(u64)),
  WIDTH(WIDTH),
  timeKernels(timeKernels),
  device(device),
  context{device},
  program(compile(args, context.get(), device, N, E, WIDTH)),
  queue(Queue::make(context, timeKernels, args.cudaYield)),

  // Specifies size in number of workgroups
#define LOAD(name, nGroups) name{program.get(), queue, device, nGroups, #name}
  // Specifies size in "work size": workSize == nGroups * groupSize
#define LOAD_WS(name, workSize) name{program.get(), queue, device, #name, workSize}

  LOAD(carryOut,   HEIGHT / 4),
  LOAD(carryIn,    HEIGHT / 4),
  LOAD(tailSquare, WIDTH),
  LOAD(tailMul,    WIDTH),
#undef LOAD_WS
#undef LOAD

  dTrigW{genSmallTrig(context, WIDTH, nW, false)},
  iTrigW{genSmallTrig(context, WIDTH, nW, true)},
  dTrigH{genSmallTrig(context, HEIGHT, nH, false)},
  iTrigH{genSmallTrig(context, HEIGHT, nH, true)},
  
  dBigTrig{genBigTrig(context, WIDTH, HEIGHT, false)},
  iBigTrig{genBigTrig(context, WIDTH, HEIGHT, true)},
  dBigTrigStep{genBigTrigStep(context, WIDTH, HEIGHT, false)},
  iBigTrigStep{genBigTrigStep(context, WIDTH, HEIGHT, true)},

  dWeights{context, "dWeights", weights.dWeights},
  iWeights{context, "iWeights", weights.iWeights},

  bufData{queue, "data", N},
  bufAux{queue, "aux", N},
  bufWords{queue, "words", N},
  bufCheck{queue, "check", N},
  bufCarry{queue, "carry", N / 2},
  buf1{queue, "buf1", N},
  buf2{queue, "buf2", N},
  buf3{queue, "buf3", N},
  args{args}
{
  carryIn.setFixedArgs(3,  dTrigW, dBigTrig, dBigTrigStep, dWeights);
  carryOut.setFixedArgs(3, iTrigW, iBigTrig, iBigTrigStep, iWeights);
  tailSquare.setFixedArgs(1, dTrigH, iTrigH);
  tailMul.setFixedArgs(2, dTrigH, iTrigH);  
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

unique_ptr<Gpu> Gpu::make(u32 E, const Args &args) {
  FFTConfig config = getFFTConfig(E, args.fftSpec);
  u32 WIDTH        = config.width;
  u32 HEIGHT       = config.height;
  u32 N = WIDTH * HEIGHT;

  u32 nW = 4;
  u32 nH = 4;

  float bitsPerWord = E / float(N);
  log("FFT: %s %s (%.2f bpw)\n", numberK(N).c_str(), config.spec().c_str(), bitsPerWord);

  if (bitsPerWord > 26) {
    log("FFT size too small for exponent (%.2f bits/word).\n", bitsPerWord);
    throw "FFT size too small";
  }

  if (bitsPerWord < 10) {
    log("FFT size too large for exponent (%.2f bits/word < %.2f bits/word).\n", bitsPerWord, 10.0);
    throw "FFT size too large";
  }

  bool timeKernels = args.timeKernels;

  return make_unique<Gpu>(args, E, WIDTH, HEIGHT, nW, nH, getDevice(args.device), timeKernels);
}

// vector<u32> Gpu::readCheck() { return readAndCompress(bufCheck); }
// vector<u32> Gpu::readData() { return readAndCompress(bufData); }

/*
bool Gpu::doCheck(u32 blockSize, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3) {
  modSqLoopMul3(bufAux, bufCheck, 0, blockSize);  
  modMul(bufCheck, bufCheck, bufData, buf1, buf2, buf3);  
  return equalNotZero(bufCheck, bufAux);
}
*/

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

u32 Gpu::modSqLoop(Buffer<i32>& wordsBuf, Buffer<i64>& carryBuf, u32 from, u32 to) {
  assert(from <= to);
  for (u32 k = from; k < to; ++k) {
    carryIn(buf1, wordsBuf, carryBuf);
    tailSquare(buf1);
    carryOut(wordsBuf, carryBuf, buf1);
  }
  return to;
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

}

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

PRPResult Gpu::isPrimePRP(const Args &args, const Task& task) {
  return {};
  
#if 0  
  u32 E = task.exponent;
  u32 b1 = task.B1;
  u32 b2 = task.B2;
  u32 k = 0, blockSize = 0;
  u32 nErrors = 0;

  log("maxAlloc: %.1f GB\n", args.maxAlloc * (1.0f / (1 << 30)));
  if (!args.maxAlloc) {
    log("You should use -maxAlloc if your GPU has more than 4GB memory. See help '-h'\n");
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
      doStop = signal.stopRequested() || (args.iters && k - startK >= args.iters);
      b1JustFinished = !b1Acc.wantK() && !didP2 && !jacobiFuture.valid() && (k - startK >= 2 * blockSize);
    }
    
    bool leadOut = doStop || b1JustFinished || (k % 10000 == 0) || (k % blockSize == 0 && k >= kEndEnd) || k == persistK || k == kEnd || useLongCarry;

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

    u64 res = dataResidue(); // implies finish()
    bool doCheck = !res || doStop || b1JustFinished || (k % checkStep == 0) || (k >= kEndEnd) || (k - startK == 2 * blockSize);
      
    if (k % 10000 == 0 && !doCheck) {
      float secsPerIt = iterationTimer.reset(k);
      // log("   %9u %6.2f%% %s %4.0f us/it\n", k, k / float(kEndEnd) * 100, hex(res).c_str(), secsPerIt * 1'000'000);
      log("%9u %s %4.0f\n", k, hex(res).c_str(), secsPerIt * 1'000'000);
    }
      
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
      if (printStats) { printRoundoff(E); }

      float secsPerIt = iterationTimer.reset(k);

      Words check = readCheck();
      if (check.empty()) { log("Check read ZERO\n"); }

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
        if (lastFailedRes64 && res == *lastFailedRes64) {
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
#endif
}

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
#include <iostream>

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

u64 weight(u32 N, u32 E, u32 H, u32 line, u32 col) {
  // return 1;
  return root2(N, extra(N, E, kAt(H, line, col)));
}

u64 invWeight(u32 N, u32 E, u32 H, u32 line, u32 col) {
  // return 1;
  return root2(N, -extra(N, E, kAt(H, line, col)));
}

Weights genWeights(u32 E, u32 W, u32 H) {
  constexpr const u64 INV_N = 0xfffffbff'00000401;
  
  u32 N = W * H;

  vector<u64> iWeights, dWeights;
  
  for (u32 line = 0; line < H; line += 4) {
    for (u32 col = 0; col < W; ++col) {
      dWeights.push_back(weight(N, E, H, line, col));
      iWeights.push_back(mul(invWeight(N, E, H, line, col), INV_N));
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
  defines.push_back({"IWSTEP_2", mul(invWeight(N, E, HEIGHT, 1, 0), 2u)});
    
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
  LOAD(sinkCarry, HEIGHT / 4),
  LOAD(tailSquare, WIDTH),
  LOAD(tailMul,    WIDTH),
  LOAD_WS(transposeWordsIn, N),
  LOAD_WS(transposeWordsOut, N),
  LOAD_WS(transposeCarryOut, N / 4),

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

  bufOut1{queue, "out1", N},
  bufOut2{queue, "out2", N},
  
  
  bufWords{queue, "words", N},
  bufCheck{queue, "check", N},
  bufCheck2{queue, "check2", N},

  bufWordsCarry{queue, "carry", N / 4},
  bufCheckCarry{queue, "carry", N / 4},
  bufCheckCarry2{queue, "carry", N / 4},
  
  buf1{queue, "buf1", N},
  buf2{queue, "buf2", N},
  
  args{args}
{
  program.reset();

  carryIn.setFixedArgs(3,  dTrigW, dBigTrig, dBigTrigStep, dWeights);
  carryOut.setFixedArgs(3, iTrigW, iBigTrig, iBigTrigStep, iWeights);
  tailSquare.setFixedArgs(1, dTrigH, iTrigH);
  tailMul.setFixedArgs(2, dTrigH, iTrigH);  
  // finish();
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

  if (bitsPerWord > 30) {
    log("FFT size too small for exponent (%.2f bits/word).\n", bitsPerWord);
    throw "FFT size too small";
  }

  if (bitsPerWord < 10) {
    log("FFT size too large for exponent (%.2f bits/word < %.2f bits/word).\n", bitsPerWord, 10.0);
    throw "FFT size too large";
  }

  bool timeKernels = args.timeKernels;
  cout << "time " << timeKernels << endl;

  return make_unique<Gpu>(args, E, WIDTH, HEIGHT, nW, nH, getDevice(args.device), timeKernels);
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

template<typename T>
void print(const string& s, const vector<T>& v, int limit = 200) {
  int cnt = 0;
  for (u32 i = 0, end = v.size(); i < end && cnt < limit; ++i) {
    if (v[i]) {
      cout << s << " " << cnt << " " << i << " " << v[i] << endl;
      ++cnt;
    }
  }
}

vector<u32> Gpu::read(const ConstBuffer<i32>& bufWords, const ConstBuffer<i64>& bufCarry, i32 mul) {
  sinkCarry(bufOut1, bufWords, bufCarry);
  transposeWordsOut(bufOut2, bufOut1);
  return compact(bufOut2.read(), E, mul);
}

void compare(const vector<u32>& a, const vector<u32>& b) {
  assert(a.size() == b.size());
  int cnt = 0;
  for (int i = 0, end = int(a.size()); i < end; ++i) {
    if (a[i] != b[i]) {
      cout << "! " << i << ' ' << a[i] << ' ' << b[i] << endl;
      ++cnt;
      assert(cnt < 20);
    }
  }
}

PRPResult Gpu::isPrimePRP(const Args &args, const Task& task) {
  vector<i32> in(N);
  
  in[0] = 3;
  bufWords.write(in);
  
  in[0] = 1;
  bufCheck.write(in);
  bufCheck2.write(in);
  
  bufWordsCarry.zero();
  bufCheckCarry.zero();
  bufCheckCarry2.zero();

  carryIn(buf1, bufWords, bufWordsCarry);
  carryIn(buf2, bufCheck, bufCheckCarry);

  for (int n = 0; n < 10; ++n) {
    for (int k = 0; k < 100; ++k) {
      tailMul(buf2, buf1);
      carryOut(bufCheck, bufCheckCarry, buf2);
      if (k == 0) {
        cout << n << endl;
        logTimeKernels();
        assert(read(bufCheck, bufCheckCarry) == read(bufCheck2, bufCheckCarry2, 3));
      }
    
      for (int i = 0; i < 200; ++i) {
        tailSquare(buf1);
        carryOut(bufWords, bufWordsCarry, buf1);
        carryIn(buf1, bufWords, bufWordsCarry);
      }
      carryIn(buf2, bufCheck, bufCheckCarry);
      queue->finish();
      // cout << '.' << endl;
    }

    for (int i = 0; i < 200; ++i) {
      tailSquare(buf2);
      carryOut(bufCheck2, bufCheckCarry2, buf2);
      carryIn(buf2, bufCheck2, bufCheckCarry2);
    }

    carryIn(buf2, bufCheck, bufCheckCarry);
  }
  
  #if 0
    // carryIn(buf1, bufWords, bufWordsCarry);
    tailSquare(buf1);
    carryOut(bufWords, bufWordsCarry, buf1);

    carryIn(buf1, bufWords, bufWordsCarry);
    tailSquare(buf1);
    carryOut(bufWords, bufWordsCarry, buf1);

    carryIn(buf1, bufCheck, bufCheckCarry);
    tailSquare(buf1);
    carryOut(bufCheck2, bufCheckCarry2, buf1);

    carryIn(buf1, bufCheck2, bufCheckCarry2);
    tailSquare(buf1);
    carryOut(bufCheck2, bufCheckCarry2, buf1);

    carryIn(buf1, bufWords, bufWordsCarry);
    carryIn(buf2, bufCheck, bufCheckCarry);
    tailMul(buf2, buf1);
    carryOut(bufCheck, bufCheckCarry, buf2);

    // compare check == 3*check2  
    auto check = read(bufCheck, bufCheckCarry);
    auto check2 = read(bufCheck2, bufCheckCarry2, 3);
    
    if (check != check2) {
      compare(check, check2);
    }
    assert(check == check2);

  for (int rep = 0; rep < 200; ++rep) {
    cout << rep << endl;
    carryIn(buf1, bufWords, bufWordsCarry);
    tailSquare(buf1);
    carryOut(bufWords, bufWordsCarry, buf1);

    carryIn(buf1, bufWords, bufWordsCarry);
    tailSquare(buf1);
    carryOut(bufWords, bufWordsCarry, buf1);

    carryIn(buf1, bufWords, bufWordsCarry);
    carryIn(buf2, bufCheck, bufCheckCarry);
    tailMul(buf2, buf1);
    carryOut(bufCheck, bufCheckCarry, buf2);

    // carryIn(buf1, bufWords, bufWordsCarry);
    tailSquare(buf1);
    carryOut(bufWords, bufWordsCarry, buf1);

    carryIn(buf1, bufWords, bufWordsCarry);
    tailSquare(buf1);
    carryOut(bufWords, bufWordsCarry, buf1);

    carryIn(buf1, bufCheck, bufCheckCarry);
    tailSquare(buf1);
    carryOut(bufCheck2, bufCheckCarry2, buf1);

    carryIn(buf1, bufCheck2, bufCheckCarry2);
    tailSquare(buf1);
    carryOut(bufCheck2, bufCheckCarry2, buf1);

    carryIn(buf1, bufWords, bufWordsCarry);
    carryIn(buf2, bufCheck, bufCheckCarry);
    tailMul(buf2, buf1);
    carryOut(bufCheck, bufCheckCarry, buf2);

    // compare check == 3*check2  
    auto check = read(bufCheck, bufCheckCarry);
    auto check2 = read(bufCheck2, bufCheckCarry2, 3);
    
    if (check != check2) {
      compare(check, check2);
    }
    assert(check == check2);
  }
  #endif
    
  return {};  
}

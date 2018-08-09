#pragma once

#include "LowGpu.h"
#include "kernel.h"
#include "state.h"
#include "args.h"
#include "common.h"

#include <cmath>

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#define TAU (2 * M_PIl)

const unsigned BUF_CONST = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
const unsigned BUF_RW    = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;

// Sets the weighting vectors direct A and inverse iA (as per IBDWT).
pair<vector<double>, vector<double>> genWeights(int E, int W, int H) {
  int N = 2 * W * H;

  vector<double> aTab, iTab;
  aTab.reserve(N);
  iTab.reserve(N);

  int baseBits = E / N;
  auto iN = 1 / (long double) N;

  for (int line = 0; line < H; ++line) {
    for (int col = 0; col < W; ++col) {
      for (int rep = 0; rep < 2; ++rep) {
        int k = (line + col * H) * 2 + rep;
        int bits  = bitlen(N, E, k);
        assert(bits == baseBits || bits == baseBits + 1);
        auto a = exp2l(extra(N, E, k) * iN);
        auto ia = 1 / (4 * N * a);
        aTab.push_back((bits == baseBits) ? a  : -a);
        iTab.push_back((bits == baseBits) ? ia : -ia);
      }
    }
  }
  assert(int(aTab.size()) == N && int(iTab.size()) == N);
  return make_pair(aTab, iTab);
}

template<typename T> struct Pair { T x, y; };

using double2 = Pair<double>;
using float2  = Pair<float>;
using uint2   = Pair<u32>;
using ulong2  = Pair<u64>;

double2 U2(double a, double b) { return double2{a, b}; }

// Returns the primitive root of unity of order N, to the power k.
template<typename T2> T2 root1(u32 N, u32 k);

template<> double2 root1<double2>(u32 N, u32 k) {
  long double angle = - TAU / N * k;
  return double2{double(cosl(angle)), double(sinl(angle))};
}

template<typename T2>
T2 *trig(T2 *p, int n, int B) {
  for (int i = 0; i < n; ++i) { *p++ = root1<T2>(B, i); }
  return p;
}

template<typename T2>
T2 *smallTrigBlock(int W, int H, T2 *p) {
  for (int line = 1; line < H; ++line) {
    for (int col = 0; col < W; ++col) {
      *p++ = root1<T2>(W * H, line * col);
    }
  }
  return p;
}

cl_mem genSmallTrig(cl_context context, int size, int radix) {
  auto *tab = new double2[size]();
  auto *p = tab + radix;
  int w = 0;
  for (w = radix; w < size; w *= radix) { p = smallTrigBlock(w, std::min(radix, size / w), p); }
  // assert(w == size);
  assert(p - tab == size);
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double2) * size, tab);
  delete[] tab;
  return buf;
}

template<typename T>
void setupWeights(cl_context context, Buffer &bufA, Buffer &bufI, int W, int H, int E) {
  int N = 2 * W * H;
  auto weights = genWeights(E, W, H);
  bufA.reset(makeBuf(context, BUF_CONST, sizeof(T) * N, weights.first.data()));
  bufI.reset(makeBuf(context, BUF_CONST, sizeof(T) * N, weights.second.data()));
}

void logTimeKernels(std::initializer_list<Kernel *> kerns) {
  struct Info {
    std::string name;
    StatsInfo stats;
  };

  double total = 0;
  std::vector<Info> infos;
  for (Kernel *k : kerns) {
    Info info{k->getName(), k->resetStats()};
    infos.push_back(info);
    total += info.stats.sum;
  }

  std::sort(infos.begin(), infos.end(), [](const Info &a, const Info &b) { return a.stats.sum >= b.stats.sum; });

  for (Info info : infos) {
    StatsInfo stats = info.stats;
    float percent = 100 / total * stats.sum;
    if (percent >= .1f) {
      log("%4.1f%% %-14s : %5.0f [%5.0f, %5.0f] us/call   x %5d calls\n",
          percent, info.name.c_str(), stats.mean, stats.low, stats.high, stats.n);
    }
  }
  log("\n");
}

// string valueDefine(const string &key, u32 value) { return key + "=" + std::to_string(value) + "u"; }

cl_device_id getDevice(const Args &args) {
  cl_device_id device = nullptr;
  if (args.device >= 0) {
    auto devices = getDeviceIDs(false);    
    assert(int(devices.size()) > args.device);
    device = devices[args.device];
  } else {
    auto devices = getDeviceIDs(true);
    if (devices.empty()) {
      log("No GPU device found. See -h for how to select a specific device.\n");
      return 0;
    }
    device = devices[0];
  }
  return device;
}

struct FftConfig {
  u32 width, height, middle;
  int fftSize;
  u32 maxExp;

  FftConfig(u32 fftSize, double maxExp, u32 width, u32 height, u32 middle) :
    width(width),
    height(height),
    middle(middle),
    fftSize(width * height * middle * 2),
    maxExp(fftSize * (17.88 + 0.36 * (24 - log2(fftSize)))) {
    assert(width  == 512 || width == 1024 || width == 2048 || width == 4096);
    assert(height == 512 || height == 1024 || height == 2048);
    assert(middle == 1 || middle == 5 || middle == 9);
  }
};

vector<FftConfig> genConfigs() {
  vector<FftConfig> configs;
  for (u32 width : {512, 1024, 2048, 4096}) {
    for (u32 height : {512, 1024, 2048}) {
      for (u32 middle : {1, 5, 9}) {
        u32 n = width * height * middle * 2;
        double maxBPW = 17.88 + 0.36 * (24 - log2(n));
        configs.push_back(FftConfig(n, n * maxBPW, width, height, middle));
      }
    }
  }
  std::sort(configs.begin(), configs.end(), [](const FftConfig &a, const FftConfig &b) {
      if (a.fftSize != b.fftSize) { return (a.fftSize < b.fftSize); }
      assert(a.width != b.width);
      if (a.width == 1024) { return true; }
      if (b.width == 1024) { return false; }
      return (a.width < b.width);
    });
  return configs;
}

FftConfig getFftConfig(const vector<FftConfig> &configs, u32 E, int argsFftSize) {
  int i = 0;
  int n = int(configs.size());
  // log("A %d %d %d\n", n, argsFftSize, E);
  if (argsFftSize < 10) { // fft delta or not specified.
    while (i < n - 1 && configs[i].maxExp < E) { ++i; }      
    i = max(0, min(i + argsFftSize, n - 1));
  } else { // user-specified fft size.
    while (i < n - 1 && argsFftSize > configs[i].fftSize) { ++i; }      
  }
  return configs[i];
}

class OpenGpu : public LowGpu<Buffer> {
  int hN, nW, nH, bufSize;
  bool useLongCarry;
  bool useMiddle;
  
  Queue queue;

  Kernel carryFused;
  Kernel fftP;
  Kernel fftW;
  Kernel fftMiddleIn;
  Kernel fftMiddleOut;
  
  Kernel carryA;
  Kernel carryM;
  Kernel shift;
  Kernel carryB;
  Kernel res36;
  Kernel compare;
  
  Kernel transposeW, transposeH;
  Kernel transposeIn, transposeOut;
  
  Kernel tailFused;
  Kernel mulFused;
  Kernel readResidue;
  
  Buffer bufGoodData, bufGoodCheck;
  Buffer bufTrigW, bufTrigH;
  Buffer bufA, bufI;
  Buffer buf1, buf2, buf3;
  Buffer bufCarry;
  Buffer bufReady;
  Buffer bufSmallOut;

  int offsetGoodData, offsetGoodCheck;

  OpenGpu(u32 E, u32 W, u32 BIG_H, u32 SMALL_H, int nW, int nH,
          cl_program program, cl_device_id device, cl_context context,
          bool timeKernels, bool useLongCarry) :
    LowGpu(E, W * BIG_H * 2),
    hN(N / 2),
    nW(nW),
    nH(nH),
    bufSize(N * sizeof(double)),
    useLongCarry(useLongCarry),
    useMiddle(BIG_H != SMALL_H),
    queue(makeQueue(device, context)),    

#define LOAD(name, workGroups) name(program, queue.get(), device, workGroups, #name, timeKernels)
    LOAD(carryFused, BIG_H + 1),
    LOAD(fftP, BIG_H),
    LOAD(fftW, BIG_H),
    LOAD(fftMiddleIn,  hN / (256 * (BIG_H / SMALL_H))),
    LOAD(fftMiddleOut, hN / (256 * (BIG_H / SMALL_H))),
    LOAD(carryA,  nW * (BIG_H/16)),
    LOAD(carryM,  nW * (BIG_H/16)),
    LOAD(shift,   nW * (BIG_H/16)),
    LOAD(carryB,  nW * (BIG_H/16)),
    LOAD(res36,   nW * (BIG_H/16)),
    LOAD(compare, nW * (BIG_H/16)),
    LOAD(transposeW,   (W/64) * (BIG_H/64)),
    LOAD(transposeH,   (W/64) * (BIG_H/64)),
    LOAD(transposeIn,  (W/64) * (BIG_H/64)),
    LOAD(transposeOut, (W/64) * (BIG_H/64)),
    LOAD(tailFused, (hN / SMALL_H) / 2),
    LOAD(mulFused,  (hN / SMALL_H) / 2),
    LOAD(readResidue, 1),
#undef LOAD
    
    bufGoodData( makeBuf(context, BUF_RW, N * sizeof(int))),
    bufGoodCheck(makeBuf(context, BUF_RW, N * sizeof(int))),    
    bufTrigW(genSmallTrig(context, W, nW)),
    bufTrigH(genSmallTrig(context, SMALL_H, nH)),
    buf1{makeBuf(    context, BUF_RW, bufSize)},
    buf2{makeBuf(    context, BUF_RW, bufSize)},
    buf3{makeBuf(    context, BUF_RW, bufSize)},
    bufCarry{makeBuf(context, BUF_RW, bufSize / 2)},
    bufReady{makeBuf(context, BUF_RW, BIG_H * sizeof(int))},
    bufSmallOut(makeBuf(context, CL_MEM_READ_WRITE, 256 * sizeof(int))),

    offsetGoodData(0), offsetGoodCheck(0)
  {
    bufData.reset( makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int)));
    bufCheck.reset(makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int)));
    bufAux.reset(  makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int)));
        
    setupWeights<double>(context, bufA, bufI, W, BIG_H, E);

    carryFused.setFixedArgs(3, bufA, bufI, bufTrigW);
    
    fftP.setFixedArgs(2, bufA, bufTrigW);
    fftW.setFixedArgs(1, bufTrigW);
    
    carryA.setFixedArgs(3, bufI);
    carryM.setFixedArgs(3, bufI);
    
    tailFused.setFixedArgs(1, bufTrigH);
    mulFused.setFixedArgs(2, bufTrigH);
    
    queue.zero(bufReady, BIG_H * sizeof(int));
  }

  std::pair<int, int> getOffsets() { return std::pair<int, int>(offsetData, offsetCheck); }
  
  vector<int> readSmall(Buffer &buf, u32 start) {
    readResidue(buf, bufSmallOut, start);
    return queue.read<int>(bufSmallOut, 128);                    
  }

  void exitKerns(Buffer &buf1, Buffer &out, bool doMul3) {    
    fftW(buf1);
    doMul3 ? carryM(buf1, out, bufCarry) : carryA(buf1, out, bufCarry);
    carryB(out, bufCarry);
  }
  
public:
  static unique_ptr<Gpu> make(u32 E, Args &args) {
    vector<FftConfig> configs = genConfigs();
    if (args.listFFT) {
      log("   FFT  maxExp %4s %4s M\n", "W", "H");
      for (auto c : configs) {
        log("%5.1fM %6.1fM %4d %4d %d\n", c.fftSize / float(1024 * 1024), c.maxExp / float(1000 * 1000), c.width, c.height, c.middle);
      }
      log("\n");
    }
        
    FftConfig config = getFftConfig(configs, E, args.fftSize);
    int WIDTH        = config.width;
    int SMALL_HEIGHT = config.height;
    int MIDDLE       = config.middle;
    int N = WIDTH * SMALL_HEIGHT * MIDDLE * 2;

    string configName = (N % (1024 * 1024)) ? std::to_string(N / 1024) + "K" : std::to_string(N / (1024 * 1024)) + "M";

    int nW = (WIDTH == 1024) ? 4 : 8;
    int nH = (SMALL_HEIGHT == 1024) ? 4 : 8;

    float bitsPerWord = E / float(N);
    string strMiddle = (MIDDLE == 1) ? "" : (string(", Middle ") + std::to_string(MIDDLE));
    log("FFT %dK: Width %d (%dx%d), Height %d (%dx%d)%s; %.2f bits/word\n",
        N / 1024, WIDTH, WIDTH / nW, nW, SMALL_HEIGHT, SMALL_HEIGHT / nH, nH, strMiddle.c_str(), bitsPerWord);

    if (bitsPerWord > 20) {
      log("FFT size too small for exponent (%.2f bits/word).\n", bitsPerWord);
      throw "FFT size too small";
    }
    
    bool useLongCarry = (bitsPerWord < 14.5f)
      || (args.carry == Args::CARRY_LONG)
      || (args.carry == Args::CARRY_AUTO && WIDTH >= 2048);
  
    log("Note: using %s carry kernels\n", useLongCarry ? "long" : "short");

    /*
    vector<string> defines {valueDefine("EXP", E),
        valueDefine("WIDTH", WIDTH),
        valueDefine("SMALL_HEIGHT", SMALL_HEIGHT),
        valueDefine("MIDDLE", MIDDLE),
        };
    */

    string clArgs = args.clArgs;
    if (!args.dump.empty()) { clArgs += " -save-temps=" + args.dump + "/" + configName; }

    bool timeKernels = args.timeKernels;
    
    cl_device_id device = getDevice(args);
    if (!device) { throw "No OpenCL device"; }

    log("%s\n", getLongInfo(device).c_str());
    if (args.cpu.empty()) { args.cpu = getShortInfo(device); }

    Context context(createContext(device));
    Holder<cl_program> program(compile(device, context.get(), "gpuowl", clArgs,
                                       {{"EXP", E}, {"WIDTH", WIDTH}, {"SMALL_HEIGHT", SMALL_HEIGHT}, {"MIDDLE", MIDDLE}}));
    if (!program) { throw "OpenCL compilation"; }

    return unique_ptr<Gpu>(new OpenGpu(E, WIDTH, SMALL_HEIGHT * MIDDLE, SMALL_HEIGHT, nW, nH,
                                       program.get(), device, context.get(), timeKernels, useLongCarry));
  }

  void finish() { queue.finish(); }
  
protected:
  void logTimeKernels() {
    ::logTimeKernels({&carryFused, &fftP, &fftW, &fftMiddleIn, &fftMiddleOut,
          &carryA, &carryM, &shift, &carryB, &res36, &compare,
          &transposeW, &transposeH, &transposeIn, &transposeOut,
          &tailFused, &mulFused, &readResidue});
  }
  
  void commit() {
    queue.copy<int>(bufData, bufGoodData, N);
    queue.copy<int>(bufCheck, bufGoodCheck, N);
    offsetGoodData  = offsetData;
    offsetGoodCheck = offsetCheck;
  }

  void rollback() {
    // Shift good data by 1.
    shift(bufGoodData, bufCarry);
    carryB(bufGoodData, bufCarry);
    offsetGoodData = (offsetGoodData + 1) % E;

    shift(bufGoodCheck, bufCarry);
    carryB(bufGoodCheck, bufCarry);
    offsetGoodCheck = (offsetGoodCheck + 1) % E;
    
    queue.copy<int>(bufGoodData, bufData, N);
    queue.copy<int>(bufGoodCheck, bufCheck, N);
    offsetData  = offsetGoodData;
    offsetCheck = offsetGoodCheck;
  }

  // Implementation of LowGpu's abstract methods below.
  
  vector<int> readOut(Buffer &buf) {
    transposeOut(buf, bufAux);
    return queue.read<int>(bufAux, N);
  }
  
  void writeIn(const vector<int> &words, Buffer &buf) {
    queue.write(bufAux, words);
    transposeIn(bufAux, buf);
  }

  void tW(Buffer &in, Buffer &out) {
    transposeW(in, out);
    if (useMiddle) { fftMiddleIn(out); }
  }

  void tH(Buffer &in, Buffer &out) {
    if (useMiddle) { fftMiddleOut(in); }
    transposeH(in, out);
  }
  
  // The IBDWT convolution squaring loop with carry propagation, on 'io', done nIters times.
  // Optional multiply-by-3 at the end.
  void modSqLoop(Buffer &in, Buffer &out, int nIters, bool doMul3) {
    assert(nIters > 0);
    
    fftP(in, buf1);
    tW(buf1, buf2);
    tailFused(buf2);
    tH(buf2, buf1);

    for (int i = 0; i < nIters - 1; ++i) {
      if (useLongCarry) {
        fftW(buf1);
        carryA(buf1, out, bufCarry);
        carryB(out, bufCarry);
        fftP(out, buf1);
      } else {
        carryFused(buf1, bufCarry, bufReady);
      }
        
      tW(buf1, buf2);
      tailFused(buf2);
      tH(buf2, buf1);
    }
    
    exitKerns(buf1, out, doMul3);
  }

  // The modular multiplication io *= in.
  void modMul(Buffer &in, Buffer &io, bool doMul3) {
    fftP(in, buf1);
    tW(buf1, buf3);
    
    fftP(io, buf1);
    tW(buf1, buf2);
    
    mulFused(buf2, buf3);

    tH(buf2, buf1);
    exitKerns(buf1, io, doMul3);
  };

  u64 reduce36(i64 x) {
    i64 r = x / (1ll << 36) + x % (1ll << 36);
    return (r < 0) ? r + (1ll << 36) - 1 : r;
  }
  
  bool equalNotZero(Buffer &buf1, u32 offset1, Buffer &buf2, u32 offset2) {
    int init[2] = {true, false};
    write(queue.get(), false, bufSmallOut, sizeof(int) * 2, init);    
    
    u32 deltaOffset = (E + offset2 - offset1) % E;
    compare(buf1, buf2, deltaOffset, bufSmallOut);
    auto readBuf = queue.read<int>(bufSmallOut, 2);
    bool isEqual   = readBuf[0];
    bool isNotZero = readBuf[1];
    bool ok = isEqual && isNotZero;

    i64 zero[2] = {0, 0};
    
    write(queue.get(), false, bufSmallOut, 2 * sizeof(i64), &zero);
    res36(buf1, offset1, bufSmallOut, 0);
    res36(buf2, offset2, bufSmallOut, 1);
    auto res = queue.read<i64>(bufSmallOut, 2);
    u64 res1 = reduce36(res[0]);
    u64 res2 = reduce36(res[1]);
    if (res1 != res2) { log("res36 differ: %09llx %09llx\n", res1, res2); }
    return ok;
  }
  
  u64 bufResidue(Buffer &buf, u32 offset) {
    u32 startWord = bitposToWord(E, N, offset);
    u32 startDword = startWord / 2;    
    u32 earlyStart = (startDword + N/2 - 32) % (N/2);
    vector<int> readBuf = readSmall(buf, earlyStart);

    u128 raw = residueFromRaw(readBuf, startWord);

    u32 startBit   = offset - wordToBitpos(E, N, startWord);
    return raw >> startBit;
  }
};
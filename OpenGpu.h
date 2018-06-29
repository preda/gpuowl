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
template<typename T>
void genWeights(int W, int H, int E, T *aTab, T *iTab) {
  T *pa = aTab;
  T *pi = iTab;

  int N = 2 * W * H;
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
        *pa++ = (bits == baseBits) ? a  : -a;
        *pi++ = (bits == baseBits) ? ia : -ia;
      }
    }
  }
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

// The generated trig table has two regions:
// - a size-H/2 quarter-circle.
// - a region of granularity TAU / (2 * W * H), used in squaring.
cl_mem genSquareTrig(cl_context context, int W, int H) {
  const int size = H/2 + W;
  auto *tab = new double2[size];
  auto *end = tab;
  end = trig(end, H/2,     H * 2);
  end = trig(end, W,     W * H * 2);
  assert(end - tab == size);
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double2) * size, tab);
  delete[] tab;
  return buf;
}

// Trig table used by transpose. Has two regions:
// - a size-2048 "full circle",
// - a region of granularity W*H and size W*H/2048.
cl_mem genTransTrig(cl_context context, int W, int H) {
  const int size = 2048 + W * H / 2048;
  auto *tab = new double2[size];
  auto *end = tab;
  end = trig(end, 2048, 2048);
  end = trig(end, W * H / 2048, W * H);
  assert(end - tab == size);
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double2) * size, tab);
  delete[] tab;
  return buf;
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
  T *aTab    = new T[N];
  T *iTab    = new T[N];
  
  genWeights(W, H, E, aTab, iTab);
  bufA.reset(makeBuf(context, BUF_CONST, sizeof(T) * N, aTab));
  bufI.reset(makeBuf(context, BUF_CONST, sizeof(T) * N, iTab));
  
  delete[] aTab;
  delete[] iTab;
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
    if (percent >= .5f) {
      log("%3.1f%% %-10s : %5.0f [%5.0f, %5.0f] us/call   x %5d calls\n",
          percent, info.name.c_str(), stats.mean, stats.low, stats.high, stats.n);
    }
  }
  log("\n");
}

string valueDefine(const string &key, u32 value) { return key + "=" + std::to_string(value) + "u"; }

cl_device_id getDevice(const Args &args) {
  cl_device_id device = nullptr;
  if (args.device >= 0) {
    cl_device_id devices[16];
    int n = getDeviceIDs(false, 16, devices);
    assert(n > args.device);
    device = devices[args.device];
  } else {
    int n = getDeviceIDs(true, 1, &device);
    if (n <= 0) {
      log("No GPU device found. See -h for how to select a specific device.\n");
      return 0;
    }
  }
  return device;
}

class OpenGpu : public LowGpu<Buffer> {
  int W, H;
  int hN, nW, nH, bufSize;
  bool useSplitTail;
  
  Queue queue;

  Kernel fftP;
  Kernel fftW;
  Kernel fftH;
  Kernel carryA;
  Kernel carryM;
  Kernel shift;
  Kernel carryB;
  Kernel transposeW;
  Kernel transposeH;
  Kernel square;
  Kernel multiply;
  Kernel tailFused;
  Kernel readResidue;
  Kernel doCheck;
  Kernel compare;
  Kernel transposeIn, transposeOut;
  
  Buffer bufGoodData, bufGoodCheck;
  Buffer bufTrigW, bufTrigH, bufTransTrig, bufSquareTrig;
  Buffer bufA, bufI;
  Buffer buf1, buf2, buf3;
  Buffer bufCarry;
  Buffer bufSmallOut;

  int offsetGoodData, offsetGoodCheck;

  OpenGpu(u32 E, u32 W, u32 H, cl_program program, cl_device_id device, cl_context context, bool timeKernels, bool useSplitTail) :
    LowGpu(E, W * H * 2),
    hN(N / 2), nW(8), nH(H / 256), bufSize(N * sizeof(double)),
    useSplitTail(useSplitTail),
    queue(makeQueue(device, context)),    

#define LOAD(name, workSize) name(program, queue.get(), device, workSize, #name, timeKernels)
    LOAD(fftP, hN / nW),
    LOAD(fftW, hN / nW),
    LOAD(fftH, hN / nH),
    LOAD(carryA, hN / 16),
    LOAD(carryM, hN / 16),
    LOAD(shift,  hN / 16),
    LOAD(carryB, hN / 16),
    LOAD(transposeW, (W/64) * (H/64) * 256),
    LOAD(transposeH, (W/64) * (H/64) * 256),
    LOAD(square,   hN / 2),
    LOAD(multiply, hN / 2),
    LOAD(tailFused, hN / (2 * nH)),
    LOAD(readResidue, 64),
    LOAD(doCheck, hN / nW),
    LOAD(compare, hN / 16),
    LOAD(transposeIn, (W/64) * (H/64) * 256),
    LOAD(transposeOut, (W/64) * (H/64) * 256),
#undef LOAD      
    
    bufGoodData( makeBuf(context, BUF_RW, N * sizeof(int))),
    bufGoodCheck(makeBuf(context, BUF_RW, N * sizeof(int))),    
    bufTrigW(genSmallTrig(context, W, nW)),
    bufTrigH(genSmallTrig(context, H, nH)),
    bufTransTrig(genTransTrig(context, W, H)),
    bufSquareTrig(genSquareTrig(context, W, H)),
    
    buf1{makeBuf(    context, BUF_RW, bufSize)},
    buf2{makeBuf(    context, BUF_RW, bufSize)},
    buf3{makeBuf(    context, BUF_RW, bufSize)},
    bufCarry{makeBuf(context, BUF_RW, bufSize)},
    bufSmallOut(makeBuf(context, CL_MEM_READ_WRITE, 256 * sizeof(int))),

    offsetGoodData(0), offsetGoodCheck(0)
  {
    bufData.reset( makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int)));
    bufCheck.reset(makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int)));
    bufAux.reset(  makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int)));
        
    setupWeights<double>(context, bufA, bufI, W, H, E);

    /*
    fftP.setArg("out", buf1);
    fftP.setArg("A", bufA);
    fftP.setArg("smallTrig", bufTrigW);
  
    fftW.setArg("io", buf1);
    fftW.setArg("smallTrig", bufTrigW);
  
    fftH.setArg("io", buf2);
    fftH.setArg("smallTrig", bufTrigH);
  
    transposeW.setArg("in",  buf1);
    transposeW.setArg("out", buf2);
    transposeW.setArg("trig", bufTransTrig);

    transposeH.setArg("in",  buf2);
    transposeH.setArg("out", buf1);
    transposeH.setArg("trig", bufTransTrig);
  
    carryA.setArg("in", buf1);
    carryA.setArg("A", bufI);
    carryA.setArg("carryOut", bufCarry);

    carryM.setArg("in", buf1);
    carryM.setArg("A", bufI);
    carryM.setArg("carryOut", bufCarry);

    // shift.setArg("io", bufData);
    shift.setArg("carryOut", bufCarry);

    carryB.setArg("carryIn", bufCarry);
  
    square.setArg("io", buf2);
    square.setArg("bigTrig", bufSquareTrig);
  
    multiply.setArg("io", buf2);
    multiply.setArg("in", buf3);
    multiply.setArg("bigTrig", bufSquareTrig);
  
    tailFused.setArg("io", buf2);
    tailFused.setArg("smallTrig", bufTrigH);
    tailFused.setArg("bigTrig", bufSquareTrig);

    doCheck.setArg("in1", bufCheck);
    doCheck.setArg("in2", bufAux);
    doCheck.setArg("out", bufSmallOut);

    compare.setArg("in1", bufCheck);
    compare.setArg("in2", bufAux);
    uint zero = 0;
    compare.setArg("offset", zero);
    compare.setArg("out", bufSmallOut);

    readResidue.setArg("in", bufData);
    readResidue.setArg("out", bufSmallOut);
    */
  }
  
public:
  static unique_ptr<Gpu> make(u32 E, Args &args) {
    int W = (E < 153'000'000) ? 2048 : 4096;
    int H = 2048;
    int N = 2 * W * H;

    string configName = (N % (1024 * 1024)) ? std::to_string(N / 1024) + "K" : std::to_string(N / (1024 * 1024)) + "M";

    int nW = 8;
    int nH = H / 256;
  
    vector<string> defines {valueDefine("EXP", E),
        valueDefine("WIDTH", W),
        valueDefine("NW", nW),
        valueDefine("HEIGHT", H),
        valueDefine("NH", nH),
        };

    bool useLongCarry = true; // args.carry == Args::CARRY_LONG || bitsPerWord < 15;  
    bool useSplitTail = args.tail == Args::TAIL_SPLIT;
  
    log("Note: using %s carry and %s tail kernels\n",
        useLongCarry ? "long" : "short", useSplitTail ? "split" : "fused");

    string clArgs = args.clArgs;
    if (!args.dump.empty()) { clArgs += " -save-temps=" + args.dump + "/" + configName; }

    bool timeKernels = args.timeKernels;
    
    cl_device_id device = getDevice(args);
    if (!device) { throw "No OpenCL device"; }

    log("%s\n", getLongInfo(device).c_str());
    if (args.cpu.empty()) { args.cpu = getShortInfo(device); }

    Context context(createContext(device));
    Holder<cl_program> program(compile(device, context.get(), "gpuowl", clArgs, defines, ""));
    if (!program) { throw "OpenCL compilation"; }

    return unique_ptr<Gpu>(new OpenGpu(E, W, H, program.get(), device, context.get(), timeKernels, useSplitTail));
  }
  

protected:
  bool equalNotZero(Buffer &buf1, Buffer &buf2, u32 deltaOffset) {    
    // compare.setArg("in1", buf1);
    // compare.setArg("in2", buf2);
    // compare.setArg("offset", deltaOffset);
    compare(buf1, buf2, deltaOffset, bufSmallOut);
    auto readBuf = queue.read<int>(bufSmallOut, 2);
    bool isEqual   = readBuf[0];
    bool isNotZero = readBuf[1];
    bool ok = isEqual && isNotZero;
    return ok;
  }
  
public:
  void finish() { queue.finish(); }
  
  void logTimeKernels() {
    ::logTimeKernels({&fftP, &fftW, &fftH, &carryA, &carryM, &carryB, &transposeW, &transposeH, &square, &multiply, &tailFused});
  }
  
  void commit() {
    queue.copy<int>(bufData, bufGoodData, N);
    queue.copy<int>(bufCheck, bufGoodCheck, N);
    offsetGoodData  = offsetData;
    offsetGoodCheck = offsetCheck;
  }

  void rollback() {
    // Shift good data by 1.
    // shift.setArg("io", bufGoodData);
    shift(bufGoodData, bufCarry);
    // carryB.setArg("io", bufGoodData);
    carryB(bufGoodData, bufCarry);
    offsetGoodData = (offsetGoodData + 1) % E;
    
    queue.copy<int>(bufGoodData, bufData, N);
    queue.copy<int>(bufGoodCheck, bufCheck, N);
    offsetData  = offsetGoodData;
    offsetCheck = offsetGoodCheck;
  }

  std::pair<int, int> getOffsets() { return std::pair<int, int>(offsetData, offsetCheck); }
  
private:
  u64 bufResidue(Buffer &buf, u32 offset) {
    u32 startWord = bitposToWord(E, N, offset);
    u32 startDword = startWord / 2;    
    u32 earlyStart = (startDword + N/2 - 32) % (N/2);
    vector<int> readBuf = readSmall(buf, earlyStart);

    u128 raw = residueFromRaw(readBuf, startWord);

    u32 startBit   = offset - wordToBitpos(E, N, startWord);
    return raw >> startBit;
  }
  
  vector<int> readSmall(Buffer &buf, u32 start) {
    // readResidue.setArg("in", buf);
    // readResidue.setArg("startDword", start);
    readResidue(buf, bufSmallOut, start);
    return queue.read<int>(bufSmallOut, 128);                    
  }
    
  vector<int> readOut(Buffer &buf) {
    // transposeOut.setArg("in", buf);
    // transposeOut.setArg("out", bufAux);
    transposeOut(buf, bufAux);
    return queue.read<int>(bufAux, N);
  }
  
  void writeIn(const vector<int> &words, Buffer &buf) {
    queue.write(bufAux, words);
    // transposeIn.setArg("in", bufAux);
    // transposeIn.setArg("out", buf);
    transposeIn(bufAux, buf);
  }
    
  // The IBDWT convolution squaring loop with carry propagation, on 'io', done nIters times.
  // Optional multiply-by-3 at the end.
  void modSqLoop(Buffer &in, Buffer &out, int nIters, bool doMul3) {
    assert(nIters > 0);
            
    entryKerns(in, buf1, buf2);
      
    // carry args needed for coreKerns.
    /*
    carryA.setArg("out", out);
    carryB.setArg("io",  out);
    fftP.setArg("in", out);
    */

    for (int i = 0; i < nIters - 1; ++i) { coreKerns(buf1, out); }

    exitKerns(buf1, out, doMul3);
  }

  // The modular multiplication io *= in.
  void modMul(Buffer &in, Buffer &io, bool doMul3) {
    directFFT(in, buf1, buf3);
    directFFT(io, buf1, buf2);
    multiply(buf2, buf3, bufSquareTrig); // input: buf2, buf3; output: buf2.
    fftH(buf2, bufTrigH);
    transposeH(buf2, buf1, bufTransTrig);
    exitKerns(buf1, io, doMul3);
  };
  
  void carry(Buffer &buf1, Buffer &bufTmp) {
    fftW(buf1, bufTrigW);
    carryA(buf1, bufI, bufTmp, bufCarry);
    carryB(bufTmp, bufCarry);
    fftP(bufTmp, buf1, bufA, bufTrigW);
  }

  void tail(Buffer &buf) {
    if (useSplitTail) {
      fftH(buf, bufTrigH);
      square(buf, bufSquareTrig);
      fftH(buf, bufTrigH);
    } else {
      tailFused(buf, bufTrigH, bufSquareTrig);
    }
  }

  void entryKerns(Buffer &in, Buffer &buf1, Buffer &buf2) {
    // fftP.setArg("in", in);      
    fftP(in, buf1, bufA, bufTrigW);
    transposeW(buf1, buf2, bufTransTrig);
    tail(buf2);
    transposeH(buf2, buf1, bufTransTrig);
  }

  void coreKerns(Buffer &buf1, Buffer &bufTmp) {
    carry(buf1, bufTmp);
    transposeW(buf1, buf2, bufTransTrig);
    tail(buf2);
    transposeH(buf2, buf1, bufTransTrig);
  }

  void exitKerns(Buffer &buf1, Buffer &out, bool doMul3) {
    // (doMul3 ? carryM : carryA).setArg("out", out);
    // carryB.setArg("io",  out);
    
    fftW(buf1, bufTrigW);
    doMul3 ? carryM(buf1, bufI, out, bufCarry) : carryA(buf1, bufI, out, bufCarry);
    carryB(out, bufCarry);
  }

  void directFFT(Buffer &in, Buffer &buf1, Buffer &out) {
    // fftP.setArg("in", in);
    // transposeW.setArg("out", out);
    // fftH.setArg("io", out);
      
    fftP(in, buf1, bufA, bufTrigW);
    transposeW(buf1, out, bufTransTrig);
    fftH(out, bufTrigH);
  }
};

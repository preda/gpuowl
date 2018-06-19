#pragma once

#include "LowGpu.h"
#include "kernel.h"
#include "state.h"
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
// - a size-H half-circle.
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

// compact 128bits from balanced uncompressed ("raw") words.
u128 residueFromRaw(int N, int E, const vector<int> &words, int startWord) {
  int start = startWord % 2 + 64;
  assert(words.size() == 128);
  assert(start == 64 || start == 65);
  int carry = 0;
  for (int i = 0; i < start; ++i) { carry = (words[i] + carry < 0) ? -1 : 0; }
  
  u128 res = 0;
  int k = startWord, hasBits = 0;
  for (auto p = words.begin() + start, end = words.end(); p < end && hasBits < 128; ++p, ++k) {
    int len = bitlen(N, E, k);
    int w = *p + carry;
    carry = (w < 0) ? -1 : 0;
    if (w < 0) { w += (1 << len); }
    assert(w >= 0 && w < (1 << len));
    res |= u128(w) << hasBits;
    hasBits += len;    
  }
  return res;
}

class OpenGpu : public LowGpu {
  int W, H;
  int hN, nW, nH, bufSize;
  bool useSplitTail;
  
  cl_context context;
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
  
  Buffer bufData, bufCheck, bufAux;
  Buffer bufGoodData, bufGoodCheck;
  Buffer bufTrigW, bufTrigH, bufTransTrig, bufSquareTrig;
  Buffer bufA, bufI;
  Buffer buf1, buf2, buf3;
  Buffer bufCarry;
  Buffer bufSmallOut;

  int offsetData, offsetCheck;
  int offsetGoodData, offsetGoodCheck;
  
public:
  OpenGpu(int E, int W, int H, cl_program program, cl_context context, cl_queue queue, cl_device_id device, bool timeKernels, bool useSplitTail) :
    LowGpu(E, W * H * 2),
    W(W),
    H(H),
    hN(N / 2), nW(8), nH(H / 256), bufSize(N * sizeof(double)),
    useSplitTail(useSplitTail),
    context(context),
    queue(queue),

#define LOAD(name, workSize) name(program, queue, device, workSize, #name, timeKernels)
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

    bufData(     makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
    bufCheck(    makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
    bufAux(      makeBuf(context, CL_MEM_READ_WRITE, N * sizeof(int))),
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

    offsetData(0), offsetCheck(0),
    offsetGoodData(0), offsetGoodCheck(0)
  {
    setupWeights<double>(context, bufA, bufI, W, H, E);

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
  }

protected:
  vector<u32> writeData(const vector<u32> &v) {
    writeIn(expandBits(v, N, E), bufData);
    offsetData = 0;
    return v;   
  }

  vector<u32> writeCheck(const vector<u32> &v) {
    writeIn(expandBits(v, N, E), bufCheck);
    offsetCheck = 0;
    return v;
  }

  vector<u32> readData()  { return compactBits(readOut(bufData),  E, offsetData); }
  vector<u32> readCheck() { return compactBits(readOut(bufCheck), E, offsetCheck); }
  
public:
  void logTimeKernels() {
    ::logTimeKernels({&fftP, &fftW, &fftH, &carryA, &carryM, &carryB, &transposeW, &transposeH, &square, &multiply, &tailFused});
  }

  void writeState(const std::vector<u32> &check, int blockSize) {
    // writeIn(expandBits(check, N, E), bufCheck);
    writeCheck(check);    

    // rebuild bufData based on bufCheck.
    modSqLoop(bufCheck, bufData, 1, false);
    for (int i = 0; i < blockSize - 2; ++i) {
      modMul(bufCheck, bufData, false);
      modSqLoop(bufData, bufData, 1, false);
    }
    modMul(bufCheck, bufData, true);

    offsetData  = 0;
    offsetCheck = 0;
  }

  /*
  std::vector<u32> roundtripData() {
    vector<u32> compact = compactBits(readOut(bufData), E, offsetData);
    writeIn(expandBits(compact, N, E), bufData);
    offsetData = 0;
    return compact;
  }
  
  std::vector<u32> roundtripCheck() {
    vector<u32> compact = compactBits(readOut(bufCheck), E, offsetCheck);
    writeIn(expandBits(compact, N, E), bufCheck);
    offsetCheck = 0;
    return compact;
  }
  */
  
  void commit() {
    queue.copy<int>(bufData, bufGoodData, N);
    queue.copy<int>(bufCheck, bufGoodCheck, N);
    offsetGoodData  = offsetData;
    offsetGoodCheck = offsetCheck;
  }

  void rollback() {
    // Shift good data by 1.
    shift.setArg("io", bufGoodData);
    shift();
    carryB.setArg("io", bufGoodData);
    carryB();
    offsetGoodData = (offsetGoodData + 1) % E;
    
    queue.copy<int>(bufGoodData, bufData, N);
    queue.copy<int>(bufGoodCheck, bufCheck, N);
    offsetData  = offsetGoodData;
    offsetCheck = offsetGoodCheck;
  }

  std::pair<int, int> getOffsets() { return std::pair<int, int>(offsetData, offsetCheck); }
  
  u64 dataResidue() { return bufResidue(bufData, offsetData); }
  
  void updateCheck() {
    modMul(bufData, bufCheck, false);
    offsetCheck = (offsetCheck + offsetData) % E;
  }
  
  // Does not change "data". Updates "check".
  bool checkAndUpdate(int blockSize) {    
    modSqLoop(bufCheck, bufAux, blockSize, true);
    u32 offsetAux = pow2(blockSize) * u64(offsetCheck) % E;
    
    updateCheck();

    u32 deltaOffset = (E + offsetAux - offsetCheck) % E;
    compare.setArg("offset", deltaOffset);
    compare();
    auto readBuf = queue.read<int>(bufSmallOut, 2);
    bool isEqual   = readBuf[0];
    bool isNotZero = readBuf[1];
    bool ok = isEqual && isNotZero;
    return ok;
  }

  void dataLoop(int reps) {
    modSqLoop(bufData, bufData, reps, false);
    offsetData = pow2(reps) * u64(offsetData) % E;
  }
  
private:
  u64 bufResidue(Buffer &buf, u32 offset) {
    u32 startWord = bitposToWord(E, N, offset);
    u32 startDword = startWord / 2;    
    u32 earlyStart = (startDword + N/2 - 32) % (N/2);
    vector<int> readBuf = readSmall(buf, earlyStart);

    u128 raw = residueFromRaw(N, E, readBuf, startWord);

    u32 startBit   = offset - wordToBitpos(E, N, startWord);
    return raw >> startBit;
  }
  
  vector<int> readSmall(Buffer &buf, u32 start) {
    readResidue.setArg("in", buf);
    readResidue.setArg("startDword", start);
    readResidue();
    return queue.read<int>(bufSmallOut, 128);                    
  }
  
  // 2**n % E
  u32 pow2(int n) {
    assert(n > 0 && n < 1024); // assert((n >> 10) == 0);
    int i = 9;
    while ((n >> i) == 0) { --i; }
    --i;
    u32 x = 2;
    for (; i >= 0; --i) {
      x = (x * u64(x)) % E;
      if (n & (1 << i)) { x = (2 * x) % E; }      
    }
    return x;
  }
  
  vector<int> readOut(Buffer &buf) {
    transposeOut.setArg("in", buf);
    transposeOut.setArg("out", bufAux);
    transposeOut();    
    return queue.read<int>(bufAux, N);
  }
  
  void writeIn(const vector<int> &words, Buffer &buf) {
    queue.write(bufAux, words);
    transposeIn.setArg("in", bufAux);
    transposeIn.setArg("out", buf);
    transposeIn();
  }
    
  // The IBDWT convolution squaring loop with carry propagation, on 'io', done nIters times.
  // Optional multiply-by-3 at the end.
  void modSqLoop(Buffer &in, Buffer &out, int nIters, bool doMul3) {
    assert(nIters > 0);
            
    entryKerns(in);
      
    // carry args needed for coreKerns.
    carryA.setArg("out", out);
    carryB.setArg("io",  out);
    fftP.setArg("in", out);

    for (int i = 0; i < nIters - 1; ++i) { coreKerns(); }

    exitKerns(out, doMul3);
  }

  // The modular multiplication io *= in.
  void modMul(Buffer &in, Buffer &io, bool doMul3) {
    directFFT(in, buf3);
    directFFT(io, buf2);
    multiply(); // input: buf2, buf3; output: buf2.
    fftH();
    transposeH();
    exitKerns(io, doMul3);
  };
  
  void carry() {
    fftW();
    carryA();
    carryB();
    fftP();
  }

  void tail() {
    if (useSplitTail) {
      fftH();
      square();
      fftH();      
    } else {
      tailFused();
    }
  }

  void entryKerns(Buffer &in) {
    fftP.setArg("in", in);
      
    fftP();
    transposeW();
    tail();
    transposeH();
  }

  void coreKerns() {
    carry();
    transposeW();
    tail();
    transposeH();
  }

  void exitKerns(Buffer &out, bool doMul3) {
    (doMul3 ? carryM : carryA).setArg("out", out);
    carryB.setArg("io",  out);
    
    fftW();
    doMul3 ? carryM() : carryA();
    carryB();
  }

  void directFFT(Buffer &in, Buffer &out) {
    fftP.setArg("in", in);
    transposeW.setArg("out", out);
    fftH.setArg("io", out);
      
    fftP();
    transposeW();
    fftH();
  }
};

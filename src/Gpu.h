// Copyright Mihai Preda.

#pragma once

#include "Buffer.h"
#include "Context.h"
#include "Queue.h"

#include "common.h"
#include "kernel.h"

#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <cmath>

struct PRPResult;
struct PRPState;
struct Task;

class Args;
class Saver;
class Signal;
class ProofSet;

using double2 = pair<double, double>;
using float2 = pair<float, float>;

namespace fs = std::filesystem;

inline u64 residue(const Words& words) { return (u64(words[1]) << 32) | words[0]; }

struct PRPResult {
  string factor;
  bool isPrime{};
  u64 res64 = 0;
  u32 nErrors = 0;
  fs::path proofPath{};
};

struct Stats {
  Stats() = default;
  Stats(float max, float mean, float sd) : max{max}, mean{mean}, sd{sd} {
    // https://en.wikipedia.org/wiki/Gumbel_distribution
    gumbelBeta = sd * 0.7797f; // sqrt(6)/pi
    gumbelMiu = mean - gumbelBeta * 0.5772f; // Euler-Mascheroni
  }

  float z(float x) { return (x - gumbelMiu) / gumbelBeta; }
  float gumbelCDF(float x) { return expf(-expf(-z(x))); }
  float gumbelRightCDF(float x) { return -expm1f(-expf(-z(x))); }

  float max, mean, sd;
  float gumbelMiu, gumbelBeta;
};

struct ROEInfo {
  u32 N;
  Stats roe;
};

class Gpu {
  u32 E;
  u32 N;

  u32 hN, nW, nH, bufSize;
  u32 WIDTH;
  bool useLongCarry;
  bool timeKernels;

  cl_device_id device;
  Context context;
  Program program;
  QueuePtr queue;
  
  Kernel kernCarryFused;
  Kernel kernCarryFusedMul;
  Kernel fftP;
  Kernel fftW;
  Kernel fftHin;
  Kernel fftHout;
  Kernel fftMiddleIn;
  Kernel fftMiddleOut;
  
  Kernel kernCarryA;
  Kernel kernCarryM;
  Kernel carryB;
  
  Kernel transposeW, transposeH;
  Kernel transposeIn, transposeOut;

  Kernel kernelMultiply;
  Kernel tailFusedSquare;
  Kernel tailFusedMulLow;
  Kernel tailFusedMul;
  Kernel tailSquareLow;
  Kernel tailMulLowLow;
  
  Kernel readResidue;
  Kernel isNotZero;
  Kernel isEqual;
  Kernel sum64;
  
  // Kernel testKernel;

  // Trigonometry constant buffers, used in FFTs.
  ConstBuffer<double2> bufTrigW;
  ConstBuffer<double2> bufTrigH;
  ConstBuffer<double2> bufTrigM;

  ConstBuffer<u32> bufBits;  // bigWord bits aligned for CarryFused/fftP
  ConstBuffer<u32> bufBitsC; // bigWord bits aligned for CarryA/M

  // "integer word" buffers. These are "small buffers": N x int.
  HostAccessBuffer<int> bufData;   // Main int buffer with the words.
  HostAccessBuffer<int> bufAux;    // Auxiliary int buffer, used in transposing data in/out and in check.
  Buffer<int> bufCheck;  // Buffers used with the error check.
  Buffer<int> bufBase;   // used in P-1 error check.

  // Carry buffers, used in carry and fusedCarry.
  Buffer<i64> bufCarry;  // Carry shuttle.
  
  Buffer<int> bufReady;  // Per-group ready flag for stairway carry propagation.

  // Small aux buffer used to read res64.
  HostAccessBuffer<int> bufSmallOut;
  HostAccessBuffer<u64> bufSumOut;

  // The round-off error ("ROE"), one float element per iteration.
  HostAccessBuffer<float> bufROE;

  // The next position to write in the ROE buffer.
  u32 roePos;

  // Auxilliary big buffers
  Buffer<double> buf1;
  Buffer<double> buf2;
  Buffer<double> buf3;
  unsigned statsBits;
  
  vector<int> readSmall(Buffer<int>& buf, u32 start);

  void tW(Buffer<double>& out, Buffer<double>& in);
  void tH(Buffer<double>& out, Buffer<double>& in);
  void tailSquare(Buffer<double>& out, Buffer<double>& in) { tailFusedSquare(out, in); }
  
  vector<int> readOut(ConstBuffer<int> &buf);
  void writeIn(Buffer<int>& buf, const vector<i32> &words);

  void coreStep(Buffer<int>& out, Buffer<int>& in, bool leadIn, bool leadOut, bool mul3);
  u32 modSqLoop(Buffer<int>& io, u32 from, u32 to);
  u32 modSqLoopMul3(Buffer<int>& out, Buffer<int>& in, u32 from, u32 to);

  bool equalNotZero(Buffer<int>& bufCheck, Buffer<int>& bufAux);
  u64 bufResidue(Buffer<int>& buf);
  
  vector<u32> writeBase(const vector<u32> &v);

  void exponentiateCore(Buffer<double>& out, const Buffer<double>& base, u64 exp, Buffer<double>& tmp);
  
  void exponentiate(Buffer<int>& bufInOut, u64 exp, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3);
  void exponentiate(Buffer<double>& out, const Buffer<double>& base, u64 exp, Buffer<double>& tmp1);
  void exponentiateLow(Buffer<double>& out, const Buffer<double>& base, u64 exp, Buffer<double>& tmp1, Buffer<double>& tmp2);

  void topHalf(Buffer<double>& out, Buffer<double>& inTmp);
  void writeState(const vector<u32> &check, u32 blockSize, Buffer<double>&, Buffer<double>&, Buffer<double>&);
  void tailMul(Buffer<double>& out, Buffer<double>& in, Buffer<double>& inTmp);
  

  Gpu(const Args& args, u32 E, u32 W, u32 BIG_H, u32 SMALL_H, u32 nW, u32 nH,
      cl_device_id device, bool timeKernels, bool useLongCarry, struct Weights&& weights);

  void printRoundoff(u32 E);

  // does either carrryFused() or the expanded version depending on useLongCarry
  void doCarry(Buffer<double>& out, Buffer<double>& in);

  void modMul(Buffer<int>& out, Buffer<int>& inA, Buffer<int>& inB, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3, bool mul3 = false);
  
  void mul(Buffer<int>& out, Buffer<int>& inA, Buffer<double>& inB, Buffer<double>& tmp1, Buffer<double>& tmp2, bool mul3 = false);

  // data := data * data;
  void square(Buffer<int>& data, Buffer<double>& tmp1, Buffer<double>& tmp2);
  
  u32 maxBuffers();

  fs::path saveProof(const Args& args, const ProofSet& proofSet);
  ROEInfo readROE();
  
  u32 updatePos(u32 bit) { return (statsBits & bit) ? roePos++ : roePos; }

public:
  const Args& args;

  template<typename... Args> void carryA(const Args &...args) { kernCarryA(updatePos(1<<2), args...); }
  void carryM(Buffer<int>& a, Buffer<double>& b) { kernCarryM(updatePos(1<<3), a, b); }
  void carryFused(Buffer<double>& a, Buffer<double>& b) { kernCarryFused(updatePos(1<<0), a, b); }
  void carryFusedMul(Buffer<double>& a, Buffer<double>& b) { kernCarryFusedMul(updatePos(1<<1), a, b);}

  void mul(Buffer<int>& out, Buffer<int>& inA, Buffer<int>& inB);
  void mul(Buffer<int>& io, Buffer<int>& inB);
  void mul(Buffer<int>& io, Buffer<double>& inB);
  void square(Buffer<int>& data);

  void finish() { queue->finish(); }

  // acc := acc * data; with "data" in lowish position.
  void accumulate(Buffer<int>& acc, Buffer<double>& data, Buffer<double>& tmp1, Buffer<double>& tmp2);

  
  static unique_ptr<Gpu> make(u32 E, const Args &args);
  static void doDiv9(u32 E, Words& words);
  static bool equals9(const Words& words);
  
  Gpu(const Args& args, u32 E, u32 W, u32 BIG_H, u32 SMALL_H, u32 nW, u32 nH,
      cl_device_id device, bool timeKernels, bool useLongCarry);

  vector<u32> readAndCompress(ConstBuffer<int>& buf);
  void writeIn(Buffer<int>& buf, const vector<u32> &words);
  void writeData(const vector<u32> &v) { writeIn(bufData, v); }
  void writeCheck(const vector<u32> &v) { writeIn(bufCheck, v); }
  
  u64 dataResidue()  { return bufResidue(bufData); }
  u64 checkResidue() { return bufResidue(bufCheck); }
    
  bool doCheck(u32 blockSize, Buffer<double>&, Buffer<double>&, Buffer<double>&);

  void logTimeKernels();

  vector<u32> readCheck();
  vector<u32> readData();

  PRPResult isPrimePRP(const Args& args, const Task& task);

  u32 getFFTSize() { return N; }

  // return A^h * B
  Words expMul(const Words& A, u64 h, const Words& B);

  // return A^h * B^2
  Words expMul2(const Words& A, u64 h, const Words& B);

  // A:= A^h * B
  void expMul(Buffer<i32>& A, u64 h, Buffer<i32>& B);
  
  // return A^(2^n)
  Words expExp2(const Words& A, u32 n);
  vector<Buffer<i32>> makeBufVector(u32 size);
};

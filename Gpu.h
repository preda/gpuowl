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
#include <variant>
#include <atomic>
#include <future>
#include <filesystem>

struct Args;
struct PRPResult;
struct PRPState;
class Saver;
class Signal;
class Task;
class ProofSet;

using double2 = pair<double, double>;

namespace fs = std::filesystem;

inline u64 residue(const Words& words) { return (u64(words[1]) << 32) | words[0]; }

struct PRPResult {
  string factor;
  bool isPrime{};
  u64 res64 = 0;
  u32 nErrors = 0;
  fs::path proofPath{};
};

struct Reload {
};

class Gpu {
  friend class SquaringSet;
  u32 E;
  u32 N;

  u32 hN, nW, nH, bufSize;
  u32 WIDTH;
  bool useLongCarry;
  bool useMergedMiddle;
  bool timeKernels;

  cl_device_id device;
  Context context;
  Holder<cl_program> program;
  QueuePtr queue;
  
  Kernel carryFused;
  Kernel carryFusedMul;
  Kernel fftP;
  Kernel fftW;
  Kernel fftHin;
  Kernel fftHout;
  Kernel fftMiddleIn;
  Kernel fftMiddleOut;
  
  Kernel carryA;
  Kernel carryM;
  Kernel carryB;
  
  Kernel transposeW, transposeH;
  Kernel transposeIn, transposeOut;

  Kernel kernelMultiply;
  Kernel kernelMultiplyDelta;
  Kernel tailFusedSquare;
  Kernel tailFusedMulDelta;
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

  // Weight constant buffers, with the direct and inverse weights. N x double.
  ConstBuffer<double> bufWeightA;      // Direct weights.
  // ConstBuffer<double> bufWeightI;      // Inverse weights.

  ConstBuffer<u32> bufBits;
  ConstBuffer<u32> bufBitsC;
  ConstBuffer<double> bufGroupWeights;
  ConstBuffer<double> bufThreadWeights;
  ConstBuffer<double> bufGroupWeightsInv;
  ConstBuffer<double> bufGroupWeightsDir;
  
  // "integer word" buffers. These are "small buffers": N x int.
  HostAccessBuffer<int> bufData;   // Main int buffer with the words.
  HostAccessBuffer<int> bufAux;    // Auxiliary int buffer, used in transposing data in/out and in check.
  Buffer<int> bufCheck;  // Buffers used with the error check.
  
  // Carry buffers, used in carry and fusedCarry.
  Buffer<i64> bufCarry;  // Carry shuttle.
  
  Buffer<int> bufReady;  // Per-group ready flag for stairway carry propagation.
  HostAccessBuffer<u32> bufRoundoff;
  HostAccessBuffer<u32> bufCarryMax;
  HostAccessBuffer<u32> bufCarryMulMax;

  // Small aux buffer used to read res64.
  HostAccessBuffer<int> bufSmallOut;
  HostAccessBuffer<u64> bufSumOut;

  // Auxilliary big buffers
  Buffer<double> buf1;
  Buffer<double> buf2;
  Buffer<double> buf3;
  
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

  // Both "io" and "in" are in "low" position
  void multiplyLowLow(Buffer<double>& io, const Buffer<double>& in, Buffer<double>& tmp);

  void exponentiateCore(Buffer<double>& out, const Buffer<double>& base, u64 exp, Buffer<double>& tmp);
  
  void exponentiate(Buffer<int>& bufInOut, u64 exp, Buffer<double>& bufBaseLow, Buffer<double>& buf1, Buffer<double>& buf2);
  void exponentiate(Buffer<double>& out, const Buffer<double>& base, u64 exp, Buffer<double>& tmp1);
  void exponentiateLow(Buffer<double>& out, const Buffer<double>& base, u64 exp, Buffer<double>& tmp1, Buffer<double>& tmp2);

  void topHalf(Buffer<double>& out, Buffer<double>& inTmp);
  void writeState(const vector<u32> &check, u32 blockSize, Buffer<double>&, Buffer<double>&, Buffer<double>&);
  void tailMulDelta(Buffer<double>& out, Buffer<double>& in, Buffer<double>& bufA, Buffer<double>& bufB);
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

  template<typename Pm1Plan>
  void doP2(Saver* saver, u32 b1, u32 b2, future<string>& gcdFuture, Signal& signal);

  void doP2(Saver* saver, u32 b1, u32 b2, future<string>& gcdFuture, Signal& signal);
  bool verifyP2Checksums(const vector<Buffer<double>>& bufs, const vector<u64>& sums);
  bool verifyP2Block(u32 D, const Words& p1Data, u32 block, const Buffer<double>& bigC, Buffer<int>& bufP2Data);
  fs::path saveProof(const Args& args, const ProofSet& proofSet);
  
public:
  const Args& args;

  Words fold(vector<Buffer<int>>& bufs);
  
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

  // std::variant<string, vector<u32>> factorPM1(u32 E, const Args& args, u32 B1, u32 B2);
  
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

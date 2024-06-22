// Copyright (C) Mihai Preda

#pragma once

#include "Background.h"
#include "Buffer.h"
#include "Context.h"
#include "Queue.h"
#include "KernelCompiler.h"

#include "Saver.h"
#include "common.h"
#include "Kernel.h"
#include "Profile.h"
#include "GpuCommon.h"
#include "FFTConfig.h"

#include <vector>
#include <memory>
#include <filesystem>
#include <cmath>

struct PRPResult;
struct Task;

class Signal;
class ProofSet;

using double2 = pair<double, double>;
using TrigBuf = Buffer<double2>;
using TrigPtr = shared_ptr<TrigBuf>;

namespace fs = std::filesystem;

inline u64 residue(const Words& words) { return (u64(words[1]) << 32) | words[0]; }

struct PRPResult {
  bool isPrime{};
  u64 res64 = 0;
  u32 nErrors = 0;
  fs::path proofPath{};
};

struct LLResult {
  bool isPrime;
  u64 res64;
};

class RoeInfo {
public:
  RoeInfo() = default;
  RoeInfo(u32 n, double max, double mean, double sd) : N{n}, max{max}, mean{mean}, sd{sd} {
    // https://en.wikipedia.org/wiki/Gumbel_distribution
    gumbelBeta = sd * 0.779696801233676; // sqrt(6)/pi
    gumbelMiu = mean - gumbelBeta * 0.577215664901533; // Euler-Mascheroni
  }

  double z(double x = 0.5) const { return N ? (x - gumbelMiu) / gumbelBeta : 0.0; }

  double gumbelCDF(double x) const { return exp(-exp(-z(x))); }
  double gumbelRightCDF(double x) const { return -expm1(-exp(-z(x))); }

  std::string toString() const;

  u32 N{};
  double max{}, mean{}, sd{};
  double gumbelMiu{}, gumbelBeta{};
};

struct Weights {
  vector<double> weightsIF;
  vector<u32> bitsCF;
  vector<u32> bitsC;
};

class Gpu {
  Queue* queue;
  Background* background;
  const Args& args;

  std::unique_ptr<Saver<PRPState>> saver;

  u32 E;
  u32 N;

  u32 WIDTH;
  u32 SMALL_H;
  u32 BIG_H;

  u32 hN, nW, nH, bufSize;
  bool useLongCarry;
  u32 wantROE{};

  Profile profile{};

  KernelCompiler compiler;
  
  Kernel kCarryFused;
  Kernel kCarryFusedROE;
  Kernel kCarryFusedMul;
  Kernel kCarryFusedMulROE;
  Kernel kCarryFusedLL;

  Kernel kCarryA;
  Kernel kCarryAROE;
  Kernel kCarryM;
  Kernel kCarryMROE;
  Kernel kCarryLL;
  Kernel carryB;

  Kernel fftP;
  Kernel fftW;

  Kernel fftHin;

  Kernel tailSquare;
  Kernel tailMul;
  Kernel tailMulLow;

  Kernel fftMidIn;
  Kernel fftMidOut;

  Kernel transpIn, transpOut;

  Kernel readResidue;
  Kernel kernIsEqual;
  Kernel sum64;

  // Kernel testKernel;

  // Twiddles: trigonometry constant buffers, used in FFTs.
  // The twiddles depend only on FFT config and do not depend on the exponent.
  TrigPtr bufTrigW;
  TrigPtr bufTrigH;
  TrigPtr bufTrigM;
  
  TrigPtr bufTrigBHW;
  TrigPtr bufTrigSquare;

  Weights weights;

  // The weights and the "bigWord bits" depend on the exponent.
  Buffer<double> bufWeights;

  Buffer<u32> bufBits;  // bigWord bits aligned for CarryFused/fftP
  Buffer<u32> bufBitsC; // bigWord bits aligned for CarryA/M

  // "integer word" buffers. These are "small buffers": N x int.
  Buffer<int> bufData;   // Main int buffer with the words.
  Buffer<int> bufAux;    // Auxiliary int buffer, used in transposing data in/out and in check.
  Buffer<int> bufCheck;  // Buffers used with the error check.
  Buffer<int> bufBase;   // used in P-1 error check.

  // Carry buffers, used in carry and fusedCarry.
  Buffer<i64> bufCarry;  // Carry shuttle.
  
  Buffer<int> bufReady;  // Per-group ready flag for stairway carry propagation.

  // Small aux buffers.
  Buffer<int> bufSmallOut;
  Buffer<u64> bufSumOut;
  Buffer<int> bufTrue;

  Buffer<float> bufROE; // The round-off error ("ROE"), one float element per iteration.
  Buffer<float> bufStatsCarry;

  u32 roePos{};   // The next position to write in the ROE stats buffer.
  u32 carryPos{}; // The next position to write in the Carry stats buffer.

  // The ROE positions originating from multiplications (as opposed to squarings).
  vector<u32> mulRoePos;

  // Auxilliary big buffers
  Buffer<double> buf1;
  Buffer<double> buf2;
  Buffer<double> buf3;

  unsigned statsBits;
  TimeInfo* timeBufVect;

  vector<int> readSmall(Buffer<int>& buf, u32 start);
  
  vector<int> readOut(Buffer<int> &buf);
  void writeIn(Buffer<int>& buf, vector<i32>&& words);

  void square(Buffer<int>& out, Buffer<int>& in, bool leadIn, bool leadOut, bool doMul3 = false, bool doLL = false);
  void squareLL(Buffer<int>& io, bool leadIn, bool leadOut) { square(io, io, leadIn, leadOut, false, true); }

  void square(Buffer<int>& io);

  u32 squareLoop(Buffer<int>& out, Buffer<int>& in, u32 from, u32 to, bool doTailMul3);
  u32 squareLoop(Buffer<int>& io, u32 from, u32 to) { return squareLoop(io, io, from, to, false); }

  bool isEqual(Buffer<int>& bufCheck, Buffer<int>& bufAux);
  u64 bufResidue(Buffer<int>& buf);
  
  vector<u32> writeBase(const vector<u32> &v);
  
  void exponentiate(Buffer<int>& bufInOut, u64 exp, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3);

  void bottomHalf(Buffer<double>& out, Buffer<double>& inTmp);

  void writeState(const vector<u32>& check, u32 blockSize);
  
  // does either carrryFused() or the expanded version depending on useLongCarry
  void doCarry(Buffer<double>& out, Buffer<double>& in);

  void mul(Buffer<int>& ioA, Buffer<double>& inB, Buffer<double>& tmp1, Buffer<double>& tmp2, bool mul3 = false);
  void mul(Buffer<int>& io, Buffer<double>& inB);

  void modMul(Buffer<int>& ioA, Buffer<int>& inB, bool mul3 = false);
  
  fs::path saveProof(const Args& args, const ProofSet& proofSet);
  std::pair<RoeInfo, RoeInfo> readROE();
  RoeInfo readCarryStats();
  
  u32 updateCarryPos(u32 bit);

  bool loadPRP(Saver<PRPState>& saver, u64& lastFailedRes64, u32& outK, u32& outBlockSize, u32& nErrors);

  vector<int> readChecked(Buffer<int>& buf);

  static void doDiv9(u32 E, Words& words);
  static bool equals9(const Words& words);

public:
  Gpu(Queue* q, GpuCommon shared, FFTConfig fft, u32 E, const vector<KeyVal>& extraConf, bool logFftSize);
  static unique_ptr<Gpu> make(Queue* q, u32 E, GpuCommon shared, FFTConfig fft,
                              const vector<KeyVal>& extraConf = {}, bool logFftSize = true);

  ~Gpu();

  PRPResult isPrimePRP(const Task& task);
  LLResult isPrimeLL(const Task& task);
  double timePRP();
  tuple<bool, u64, RoeInfo, RoeInfo> measureROE(bool quick);

  Saver<PRPState> *getSaver();

  void carryA(Buffer<double>& a, Buffer<double>& b) { carryA(reinterpret_cast<Buffer<int>&>(a), b); }

  void carryA(Buffer<int>& a, Buffer<double>& b);

  void carryM(Buffer<int>& a, Buffer<double>& b);

  void carryLL(Buffer<int>& a, Buffer<double>& b);

  void carryFused(Buffer<double>& a, Buffer<double>& b);

  void carryFusedMul(Buffer<double>& a, Buffer<double>& b);

  void carryFusedLL(Buffer<double>& a, Buffer<double>& b)  { kCarryFusedLL(updateCarryPos(1<<0), a, b);}

  void writeIn(Buffer<int>& buf, const vector<u32> &words);
  
  u64 dataResidue()  { return bufResidue(bufData); }
  u64 checkResidue() { return bufResidue(bufCheck); }
    
  bool doCheck(u32 blockSize);

  void logTimeKernels();

  Words readAndCompress(Buffer<int>& buf);
  vector<u32> readCheck();
  vector<u32> readData();


  u32 getFFTSize() { return N; }

  // return A^h * B
  Words expMul(const Words& A, u64 h, const Words& B, bool doSquareB);

  // return A^h * B^2
  Words expMul2(const Words& A, u64 h, const Words& B);

  // A:= A^h * B
  void expMul(Buffer<i32>& A, u64 h, Buffer<i32>& B);
  
  // return A^(2^n)
  Words expExp2(const Words& A, u32 n);
  vector<Buffer<i32>> makeBufVector(u32 size);
private:
  u32 getProofPower(u32 k);
  void doBigLog(u32 k, u64 res, bool checkOK, float secsPerIt, u32 nIters, u32 nErrors);
};

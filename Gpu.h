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

struct Args;
struct PRPResult;
struct PRPState;

using double2 = pair<double, double>;

class Gpu {
  friend class SquaringSet;
  u32 E;
  u32 N;

  u32 hN, nW, nH, bufSize;
  bool useLongCarry;
  bool useMiddle;
  bool timeKernels;

  cl_device_id device;
  Context context;
  Holder<cl_program> program;
  QueuePtr queue;
  
  Kernel carryFused;
  Kernel carryFusedMul;
  Kernel fftP;
  Kernel fftW;
  Kernel fftH;
  Kernel fftMiddleIn;
  Kernel fftMiddleOut;
  
  Kernel carryA;
  Kernel carryM;
  Kernel carryB;
  
  Kernel transposeW, transposeH;
  Kernel transposeIn, transposeOut;

  Kernel multiply;
  Kernel square;
  Kernel tailFused;
  Kernel tailFusedMulDelta;
  
  Kernel readResidue;
  Kernel isNotZero;
  Kernel isEqual;
  Kernel sum64;

  // Trigonometry constant buffers, used in FFTs.
  ConstBuffer<double2> bufTrigW;
  ConstBuffer<double2> bufTrigH; 

  // Weight constant buffers, with the direct and inverse weights. N x double.
  ConstBuffer<double> bufWeightA;      // Direct weights.
  ConstBuffer<double> bufWeightI;      // Inverse weights.

  ConstBuffer<u32> bufBits;
  ConstBuffer<u32> bufExtras;
  ConstBuffer<double> bufGroupWeights;
  ConstBuffer<double> bufThreadWeights;
  
  // "integer word" buffers. These are "small buffers": N x int.
  HostAccessBuffer<int> bufData;   // Main int buffer with the words.
  HostAccessBuffer<int> bufAux;    // Auxiliary int buffer, used in transposing data in/out and in check.
  Buffer<int> bufCheck;  // Buffers used with the error check.
  
  // Carry buffers, used in carry and fusedCarry.
  Buffer<i64> bufCarry;  // Carry shuttle.
  
  Buffer<int> bufReady;  // Per-group ready flag for stairway carry propagation.

  // Small aux buffer used to read res64.
  HostAccessBuffer<int> bufSmallOut;
  HostAccessBuffer<u64> bufSumOut;

  vector<u32> computeBase(u32 E, u32 B1);
  pair<vector<u32>, vector<u32>> seedPRP(u32 E, u32 B1);
  
  vector<int> readSmall(Buffer<int>& buf, u32 start);

  void tW(Buffer<double>& in, Buffer<double>& out);
  void tH(Buffer<double>& in, Buffer<double>& out);
  
  vector<int> readOut(ConstBuffer<int> &buf);
  void writeIn(const vector<u32> &words, Buffer<int>& buf);
  void writeIn(const vector<int> &words, Buffer<int>& buf);
  
  void modSqLoop(u32 reps, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<int>& io, bool mul3 = false);
  
  void modMul(Buffer<int>& in, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3, Buffer<int>& io, bool mul3 = false);
  bool equalNotZero(Buffer<int>& bufCheck, Buffer<int>& bufAux);
  u64 bufResidue(Buffer<int>& buf);
  
  vector<u32> writeBase(const vector<u32> &v);

  PRPState loadPRP(u32 E, u32 iniBlockSize, Buffer<double>&, Buffer<double>&, Buffer<double>&);

  void coreStep(bool leadIn, bool leadOut, bool mul3, Buffer<double>& buf1, Buffer<double>& buf2, Buffer<int>& io);

  void multiplyLow(Buffer<double>& in, Buffer<double>& tmp, Buffer<double>& io);
  void exponentiate(const Buffer<double>& base, u64 exp, Buffer<double>& tmp, Buffer<double>& out);
  void topHalf(Buffer<double>& tmp, Buffer<double>& io);
  void writeState(const vector<u32> &check, u32 blockSize, Buffer<double>&, Buffer<double>&, Buffer<double>&);

  Gpu(const Args& args, u32 E, u32 W, u32 BIG_H, u32 SMALL_H, u32 nW, u32 nH,
      cl_device_id device, bool timeKernels, bool useLongCarry, struct Weights&& weights);

  vector<u32> readAndCompress(ConstBuffer<int>& buf);
  
public:
  static unique_ptr<Gpu> make(u32 E, const Args &args);
  
  Gpu(const Args& args, u32 E, u32 W, u32 BIG_H, u32 SMALL_H, u32 nW, u32 nH,
      cl_device_id device, bool timeKernels, bool useLongCarry);

  vector<u32> roundtripData()  { return readData(); }
  vector<u32> roundtripCheck() { return readCheck(); }

  vector<u32> writeData(const vector<u32> &v);
  vector<u32> writeCheck(const vector<u32> &v);
  
  u64 dataResidue()  { return bufResidue(bufData); }
  u64 checkResidue() { return bufResidue(bufCheck); }
    
  bool doCheck(u32 blockSize, Buffer<double>&, Buffer<double>&, Buffer<double>&);
  void updateCheck(Buffer<double>& buf1, Buffer<double>& buf2, Buffer<double>& buf3);

  void finish();

  void logTimeKernels();

  vector<u32> readCheck() { return readAndCompress(bufCheck); }
  vector<u32> readData() { return readAndCompress(bufData); }

  std::tuple<bool, u64, u32> isPrimePRP(u32 E, const Args& args);

  void buildProof(u32 E, const Args& args);
  
  std::variant<string, vector<u32>> factorPM1(u32 E, const Args& args, u32 B1, u32 B2);
  
  u32 getFFTSize() { return N; }
};

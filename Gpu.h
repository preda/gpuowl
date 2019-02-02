// Copyright Mihai Preda.

#pragma once

#include "clwrap.h"
#include "common.h"
#include "kernel.h"

#include <vector>
#include <string>
#include <memory>

struct Args;
struct PRPResult;
struct PRPState;

class Gpu {
  u32 E;
  u32 N;

  int hN, nW, nH, bufSize;
  bool useLongCarry;
  bool useMiddle;

  cl_device_id device;
  Context context;
  Holder<cl_program> program;
  Queue queue;
  
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

  // Trigonometry constant buffers, used in FFTs.
  Buffer bufTrigW;
  Buffer bufTrigH; 

  // Weight constant buffers, with the direct and inverse weights. N x double.
  Buffer bufWeightA;      // Direct weights.
  Buffer bufWeightI;      // Inverse weights.

  // "integer word" buffers. These are "small buffers": N x int.
  Buffer bufData;   // Main int buffer with the words.
  Buffer bufAux;    // Auxiliary int buffer, used in transposing data in/out and in check.
  Buffer bufCheck;  // Buffers used with the error check.
  
  // "work temporary buffers". These are "big buffers": N x double.
  // Buffer buf1, buf2, buf3;

  // Carry buffers, used in carry and fusedCarry.
  Buffer bufCarry;  // Carry shuttle.
  Buffer bufReady;  // Per-group ready flag for starway carry propagation.

  // Small aux buffer used to read res64.
  Buffer bufSmallOut;

  vector<u32> computeBase(u32 E, u32 B1);
  pair<vector<u32>, vector<u32>> seedPRP(u32 E, u32 B1);
  
  vector<int> readSmall(Buffer &buf, u32 start);

  void tW(Buffer &in, Buffer &out);
  void tH(Buffer &in, Buffer &out);
  
  void copyFromTo(Buffer &from, Buffer &to);
  
  vector<int> readOut(Buffer &buf);
  void writeIn(const vector<u32> &words, Buffer &buf);
  void writeIn(const vector<int> &words, Buffer &buf);
  
  void modSqLoop(u32 reps, bool mul3, Buffer& buf1, Buffer& buf2, Buffer &io);
  
  void modMul(Buffer &in, bool mul3, Buffer& buf1, Buffer& buf2, Buffer& buf3, Buffer& io);
  bool equalNotZero(Buffer &bufCheck, Buffer &bufAux);
  u64 bufResidue(Buffer &buf);
  
  vector<u32> writeBase(const vector<u32> &v);

  PRPState loadPRP(u32 E, u32 iniBlockSize, Buffer&, Buffer&, Buffer&);

  void coreStep(bool leadIn, bool leadOut, bool mul3, Buffer& buf1, Buffer& buf2, Buffer &io);

  void multiplyLow(Buffer& in, Buffer& tmp, Buffer& io);
  void exponentiate(Buffer& base, u64 exp, Buffer& tmp, Buffer& out);
  void topHalf(Buffer& tmp, Buffer& io);
  void writeState(const vector<u32> &check, u32 blockSize, Buffer&, Buffer&, Buffer&);
  
public:
  static unique_ptr<Gpu> make(u32 E, const Args &args);
  
  Gpu(const Args& args, u32 E, u32 W, u32 BIG_H, u32 SMALL_H, int nW, int nH,
      cl_device_id device, bool timeKernels, bool useLongCarry);


  
  vector<u32> roundtripData()  { return writeData(readData()); }
  vector<u32> roundtripCheck() { return writeCheck(readCheck()); }

  vector<u32> writeData(const vector<u32> &v);
  vector<u32> writeCheck(const vector<u32> &v);
  
  u64 dataResidue()  { return bufResidue(bufData); }
  u64 checkResidue() { return bufResidue(bufCheck); }
    
  bool doCheck(int blockSize, Buffer&, Buffer&, Buffer&);
  void updateCheck(Buffer& buf1, Buffer& buf2, Buffer& buf3);

  // void dataLoop(u32 reps) { modSqLoop(bufData, reps); }
  
  void finish();

  void logTimeKernels();

  vector<u32> readCheck();
  vector<u32> readData();

  std::pair<bool, u64> isPrimePRP(u32 E, const Args &args);
  string factorPM1(u32 E, const Args& args, u32 B1, u32 B2);
  
  u32 getFFTSize() { return N; }
};

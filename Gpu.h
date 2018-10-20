// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "clwrap.h"
#include "common.h"

#include <vector>
#include <string>
#include <memory>
#include <unordered_set>

struct Args;
class GCD;

class Gpu {
protected:
  u32 E;
  u32 N;

  Buffer bufData, bufCheck, bufAux, bufBase, bufAcc;
  
  unique_ptr<GCD> gcd;

  vector<u32> computeBase(u32 E, u32 B1);
  pair<vector<u32>, vector<u32>> seedPRP(u32 E, u32 B1);
  
protected:
  virtual void copyFromTo(Buffer &from, Buffer &to) = 0;
  
  virtual vector<int> readOut(Buffer &buf) = 0;
  virtual void writeIn(const vector<int> &words, Buffer &buf) = 0;
  
  virtual void modSqLoopMul(Buffer &io, const vector<bool> &muls) = 0;
  virtual void modSqLoopAcc(Buffer &io, const vector<bool> &muls) = 0;
  
  virtual void modMul(Buffer &in, Buffer &io) = 0;
  virtual bool equalNotZero(Buffer &bufCheck, Buffer &bufAux) = 0;
  virtual u64 bufResidue(Buffer &buf) = 0;
  
  virtual vector<u32> writeBase(const vector<u32> &v) = 0;

  // u64 residueFromRaw(const vector<int> &words);
  
public:
  Gpu(u32 E, u32 N);
  virtual ~Gpu();

  void writeState(const vector<u32> &check, const vector<u32> &base, u32 blockSize);
  
  vector<u32> roundtripData()  { return writeData(readData()); }
  vector<u32> roundtripCheck() { return writeCheck(readCheck()); }

  vector<u32> writeData(const vector<u32> &v);
  vector<u32> writeCheck(const vector<u32> &v);
  
  u64 dataResidue();
  u64 checkResidue();
    
  bool doCheck(int blockSize);
  void updateCheck();

  // returns nb. of Ks selected for GCD accumulation.
  u32 dataLoopAcc(u32 kBegin, u32 kEnd, const unordered_set<u32> &kset);
  void dataLoopMul(const vector<bool> &muls);
  
  virtual void finish() = 0;

  virtual void logTimeKernels() = 0;

  vector<u32> readCheck();
  vector<u32> readData();
  vector<u32> readAcc();

  bool isPrimePRP(u32 E, const Args &args, u32 B1, u64 *outRes, u64 *outBaseRes, string *outFactor);
  u32 getFFTSize() { return N; }
};

// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"

#include <vector>
#include <string>
#include <memory>
#include <unordered_set>

struct Args;
class GCD;

class Gpu {
  unique_ptr<GCD> gcd;

  vector<u32> computeBase(u32 E, u32 B1);
  pair<vector<u32>, vector<u32>> seedPRP(u32 E, u32 B1);
  
protected:
  virtual vector<u32> readCheck() = 0;
  virtual vector<u32> writeCheck(const vector<u32> &v) = 0;
  
public:
  Gpu();
  virtual ~Gpu();

  virtual void writeState(const vector<u32> &check, const vector<u32> &base, u32 blockSize) = 0;
  
  vector<u32> roundtripData()  { return writeData(readData()); }
  vector<u32> roundtripCheck() { return writeCheck(readCheck()); }
  virtual vector<u32> writeData(const vector<u32> &v) = 0;
  
  virtual u64 dataResidue() = 0;
  virtual u64 checkResidue() = 0;
    
  virtual bool doCheck(int blockSize) = 0;
  virtual void updateCheck() = 0;

  // returns nb. of Ks selected for GCD accumulation.
  virtual u32 dataLoopAcc(u32 kBegin, u32 kEnd, const unordered_set<u32> &kset) = 0;
  virtual void dataLoopMul(const vector<bool> &muls) = 0;
  
  virtual void finish() = 0;

  virtual u32 getFFTSize() = 0;
  virtual void logTimeKernels() = 0;

  virtual vector<u32> readData() = 0; // Used directly only by PM1 (PRP uses roundtripData()).

  virtual vector<u32> readAcc() = 0;

  bool isPrimePRP(u32 E, const Args &args, u32 B1, u64 *outRes, u64 *outBaseRes, string *outFactor);
};

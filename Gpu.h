// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"

#include <vector>
#include <string>

struct Args;

class Gpu {
protected:
  virtual vector<u32> readCheck() = 0;
  virtual vector<u32> writeCheck(const vector<u32> &v) = 0;
  
public:
  virtual ~Gpu();

  virtual void writeState(const vector<u32> &check, const vector<u32> &base, int blockSize) = 0;
  
  vector<u32> roundtripData()  { return writeData(readData()); }
  vector<u32> roundtripCheck() { return writeCheck(readCheck()); }
  virtual vector<u32> writeData(const vector<u32> &v) = 0;
  
  virtual u64 dataResidue() = 0;
  virtual u64 checkResidue() = 0;
    
  virtual void startCheck(int blockSize) = 0;
  virtual bool finishCheck() = 0;
  virtual void updateCheck() = 0;

  bool checkAndUpdate(int blockSize) {
    startCheck(blockSize);
    return finishCheck();
  }
  
  virtual void dataLoop(int reps) = 0;
  virtual void dataLoop(const vector<bool> &muls) = 0;
  
  virtual void finish() = 0;

  virtual u32 getFFTSize() = 0;
  virtual void logTimeKernels() = 0;

  virtual vector<u32> readData() = 0; // Used directly only by PM1 (PRP uses roundtripData()).

  virtual void gcdAccumulate(bool isFirst) = 0;
  virtual vector<u32> readAcc() = 0;

  string factorPM1(u32 E, u32 taskB1, const Args &args);
  bool isPrimePRP(u32 E, const Args &args, u64 *res64, u32 *nErrors, u32 *fftSize);
  bool isPrimePRPF(u32 E, u32 B1, const Args &args, u64 *res64, string *factor);
};

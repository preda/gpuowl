#pragma once

#include "common.h"

#include <vector>

class Gpu {
protected:
  virtual vector<u32> readData() = 0;
  virtual vector<u32> readCheck() = 0;
  virtual vector<u32> writeData(const vector<u32> &v) = 0;
  virtual vector<u32> writeCheck(const vector<u32> &v) = 0;
  
public:
  virtual ~Gpu() {};

  virtual void writeState(const vector<u32> &check, int blockSize) = 0;
  
  vector<u32> roundtripData()  { return writeData(readData()); }
  vector<u32> roundtripCheck() { return writeCheck(readCheck()); }

  virtual u64 dataResidue() = 0;
  virtual u64 checkResidue() = 0;
  
  virtual bool checkAndUpdate(int blockSize) = 0;
  virtual void updateCheck() = 0;
  virtual void dataLoop(int reps) = 0;
  virtual void finish() = 0;

  virtual u32 getFFTSize() = 0;
  virtual void logTimeKernels() = 0;
};

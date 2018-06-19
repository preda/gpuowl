#pragma once

#include "common.h"

#include <vector>

class LowGpu {
  u32 E; // exponent.
  u32 N; // FFT size.
  
public:
  LowGpu(u32 E, u32 N) : E(E), N(N) {}
  
  virtual ~LowGpu();

  virtual void writeState(const vector<u32> &check, int blockSize) = 0;
  
  virtual void commit() = 0;   // Make a copy for rollback() to revert to.
  virtual void rollback() = 0; // Revert to the state at most recent commit().

  vector<u32> roundtripData()  { return writeData(readData()); }
  vector<u32> roundtripCheck() { return writeCheck(readCheck()); }

  virtual u64 dataResidue() = 0;

  virtual bool checkAndUpdate(int blockSize) = 0;
  virtual void updateCheck() = 0;
  virtual void dataLoop(int reps) = 0;
  
protected:
  virtual vector<u32> readData() = 0;
  virtual vector<u32> readCheck() = 0;
  virtual vector<u32> writeData(const vector<u32> &v) = 0;
  virtual vector<u32> writeCheck(const vector<u32> &v) = 0;
};

#pragma once

#include "Gpu.h"
#include "state.h"

template<typename Buffer>
class LowGpu : public Gpu {
protected:
  u32 E; // exponent.
  u32 N; // FFT size.

  Buffer bufData, bufCheck, bufAux;

  virtual vector<int> readOut(Buffer &buf) = 0;
  virtual void writeIn(const vector<int> &words, Buffer &buf) = 0;
  
  virtual void modSqLoop(Buffer &in, Buffer &out, int nIters, bool doMul3) = 0;
  virtual void modSqLoop(Buffer &in, Buffer &out, const vector<bool> &muls) = 0;
  
  virtual void modMul(Buffer &in, Buffer &io, bool doMul3) = 0;
  virtual bool equalNotZero(Buffer &bufCheck, Buffer &bufAux) = 0;
  virtual u64 bufResidue(Buffer &buf) = 0;
  
  vector<u32> readData()  { return compactBits(readOut(bufData),  E); }
  vector<u32> readCheck() { return compactBits(readOut(bufCheck), E); }

  vector<u32> writeData(const vector<u32> &v) {
    writeIn(expandBits(v, N, E), bufData);
    return v;
  }

  vector<u32> writeCheck(const vector<u32> &v) {
    writeIn(expandBits(v, N, E), bufCheck);
    return v;
  }

  // compact 64bits from balanced uncompressed ("raw") words.
  u64 residueFromRaw(const vector<int> &words) {
    assert(words.size() == 128);
    int carry = 0;
    for (int i = 0; i < 64; ++i) { carry = (words[i] + carry < 0) ? -1 : 0; }
    
    u64 res = 0;
    int k = 0, hasBits = 0;
    for (auto p = words.begin() + 64, end = words.end(); p < end && hasBits < 64; ++p, ++k) {
      u32 len = bitlen(N, E, k);
      int w = *p + carry;
      carry = (w < 0) ? -1 : 0;
      if (w < 0) { w += (1 << len); }
      assert(w >= 0 && w < (1 << len));
      res |= u64(w) << hasBits;
      hasBits += len;
    }
    return res;
  }

public:
  LowGpu(u32 E, u32 N) : E(E), N(N) {}

  u64 dataResidue() { return bufResidue(bufData); }
  u64 checkResidue() { return bufResidue(bufCheck); }
  
  void writeState(const vector<u32> &check, int blockSize) {
    writeCheck(check);

    // rebuild bufData based on bufCheck.
    modSqLoop(bufCheck, bufData, 1, false);
    
    for (int i = 0; i < blockSize - 2; ++i) {
      modMul(bufCheck, bufData, false);
      modSqLoop(bufData, bufData, 1, false);
    }
    modMul(bufCheck, bufData, true);
  }

  void updateCheck() { modMul(bufData, bufCheck, false); }
  
  bool checkAndUpdate(int blockSize) {
    modSqLoop(bufCheck, bufAux, blockSize, true);
    updateCheck();
    return equalNotZero(bufCheck, bufAux);
  }

  void dataLoop(int reps) { modSqLoop(bufData, bufData, reps, false); }
  void dataLoop(const vector<bool> &muls) { modSqLoop(bufData, bufData, muls); }
  u32 getFFTSize() { return N; }
};

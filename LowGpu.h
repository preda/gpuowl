// gpuOwl, a Mersenne primality tester.
// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "Gpu.h"
#include "state.h"

template<typename Buffer>
class LowGpu : public Gpu {
protected:
  u32 E; // exponent.
  u32 N; // FFT size.

  Buffer bufData, bufCheck, bufAux, bufBase, bufAcc;

  virtual void copyFromTo(Buffer &from, Buffer &to) = 0;
  
  virtual vector<int> readOut(Buffer &buf) = 0;
  virtual void writeIn(const vector<int> &words, Buffer &buf) = 0;
  
  virtual void modSqLoopAcc(Buffer &in, Buffer &out, int nIters, bool doAcc) = 0;
  virtual void modSqLoop(Buffer &in, Buffer &out, const vector<bool> &muls) = 0;
  
  virtual void modMul(Buffer &in, Buffer &io, bool doMul3) = 0;
  virtual bool equalNotZero(Buffer &bufCheck, Buffer &bufAux) = 0;
  virtual u64 bufResidue(Buffer &buf) = 0;
  
  vector<u32> readData()  override { return compactBits(readOut(bufData),  E); }
  vector<u32> readCheck() override { return compactBits(readOut(bufCheck), E); }
  vector<u32> readAcc()   override { return compactBits(readOut(bufAcc), E); }

  vector<u32> writeData(const vector<u32> &v) {
    writeIn(expandBits(v, N, E), bufData);
    return v;
  }

  vector<u32> writeCheck(const vector<u32> &v) {
    writeIn(expandBits(v, N, E), bufCheck);
    return v;
  }

  virtual vector<u32> writeBase(const vector<u32> &v) = 0;

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
  
  void writeState(const vector<u32> &check, const vector<u32> &base, u32 blockSize) override {
    assert(blockSize > 0);
    
    writeCheck(check);
    copyFromTo(bufCheck, bufData);        
    copyFromTo(bufCheck, bufBase);

    u32 n = 0;
    for (n = 1; blockSize % (2 * n) == 0; n *= 2) {
      modSqLoopAcc(bufData, bufData, n, false);
      modMul(bufBase, bufData, false);
      copyFromTo(bufData, bufBase);
    }

    assert((n & (n - 1)) == 0);
    assert(blockSize % n == 0);
    
    blockSize /= n;
    for (u32 i = 0; i < blockSize - 1; ++i) {
      modSqLoopAcc(bufData, bufData, n, false);
      modMul(bufBase, bufData, false);
    }
    
    writeBase(base);
    modMul(bufBase, bufData, false);
  }

  void updateCheck() { modMul(bufData, bufCheck, false); }
  
  void startCheck(int blockSize) override {
    modSqLoopAcc(bufCheck, bufAux, blockSize, false);
    modMul(bufBase, bufAux, false);
    updateCheck();
  }

  // invoked after startCheck() !
  bool finishCheck() override { return equalNotZero(bufCheck, bufAux); }

  void dataLoop(int reps, bool doAcc) override { modSqLoopAcc(bufData, bufData, reps, doAcc); }
  void dataLoop(const vector<bool> &muls) override { modSqLoop(bufData, bufData, muls); }
  u32 getFFTSize() override { return N; }
};

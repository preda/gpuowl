#pragma once

#include "Gpu.h"
#include "state.h"

// 2**n % E
u32 pow2(u32 E, int n) {
  assert(n > 0 && n < 1024); // assert((n >> 10) == 0);
  int i = 9;
  while ((n >> i) == 0) { --i; }
  --i;
  u32 x = 2;
  for (; i >= 0; --i) {
    x = (x * u64(x)) % E;
    if (n & (1 << i)) { x = (2 * x) % E; }      
  }
  return x;
}

template<typename Buffer>
class LowGpu : public Gpu {
protected:
  u32 E; // exponent.
  u32 N; // FFT size.

  Buffer bufData, bufCheck, bufAux;
  int offsetData, offsetCheck;

  virtual vector<int> readOut(Buffer &buf) = 0;
  virtual void writeIn(const vector<int> &words, Buffer &buf) = 0;
  
  virtual void modSqLoop(Buffer &in, Buffer &out, int nIters, bool doMul3) = 0;
  virtual void modMul(Buffer &in, Buffer &io, bool doMul3) = 0;
  virtual bool equalNotZero(Buffer &bufCheck, Buffer &bufAux, u32 deltaOffset) = 0;
  virtual u64 bufResidue(Buffer &buf, u32 offset) = 0;
  
  vector<u32> readData()  { return compactBits(readOut(bufData),  E, offsetData); }
  vector<u32> readCheck() { return compactBits(readOut(bufCheck), E, offsetCheck); }

  vector<u32> writeData(const vector<u32> &v) {
    writeIn(expandBits(v, N, E), bufData);
    offsetData = 0;
    return v;
  }

  vector<u32> writeCheck(const vector<u32> &v) {
    writeIn(expandBits(v, N, E), bufCheck);
    offsetCheck = 0;
    return v;
  }

  // compact 128bits from balanced uncompressed ("raw") words.
  u128 residueFromRaw(const vector<int> &words, int startWord) {
    int start = startWord % 2 + 64;
    assert(words.size() == 128);
    assert(start == 64 || start == 65);
    int carry = 0;
    for (int i = 0; i < start; ++i) { carry = (words[i] + carry < 0) ? -1 : 0; }
    
    u128 res = 0;
    int k = startWord, hasBits = 0;
    for (auto p = words.begin() + start, end = words.end(); p < end && hasBits < 128; ++p, ++k) {
      int len = bitlen(N, E, k);
      int w = *p + carry;
      carry = (w < 0) ? -1 : 0;
      if (w < 0) { w += (1 << len); }
      assert(w >= 0 && w < (1 << len));
      res |= u128(w) << hasBits;
      hasBits += len;    
    }
    return res;
  }

public:
  LowGpu(u32 E, u32 N) : E(E), N(N), offsetData(0), offsetCheck(0) {}

  u64 dataResidue() { return bufResidue(bufData, offsetData); }
  u64 checkResidue() { return bufResidue(bufCheck, offsetCheck); }
  
  void writeState(const vector<u32> &check, int blockSize) {
    writeCheck(check);
    // log("check res %016llx %016llx\n", dataResidue(), checkResidue());

    // rebuild bufData based on bufCheck.
    modSqLoop(bufCheck, bufData, 1, false);
    // log("A %016llx %016llx\n", dataResidue(), checkResidue());
    
    for (int i = 0; i < blockSize - 2; ++i) {
      modMul(bufCheck, bufData, false);
      modSqLoop(bufData, bufData, 1, false);
    }
    modMul(bufCheck, bufData, true);

    offsetData  = 0;
    offsetCheck = 0;
  }

  void updateCheck() {
    // log("updateCheck data %016llx\n", dataResidue());
    modMul(bufData, bufCheck, false);
    offsetCheck = (offsetCheck + offsetData) % E;
  }
  
  bool checkAndUpdate(int blockSize) {
    // log("check1 %016llx\n", bufResidue(bufCheck, 0));
    modSqLoop(bufCheck, bufAux, blockSize, true);
    u32 offsetAux = pow2(E, blockSize) * u64(offsetCheck) % E;

    // log("aux1 %016llx\n", bufResidue(bufAux, 0));
    
    updateCheck();
    u32 deltaOffset = (E + offsetAux - offsetCheck) % E;
    return equalNotZero(bufCheck, bufAux, deltaOffset);
  }

  void dataLoop(int reps) {
    modSqLoop(bufData, bufData, reps, false);
    offsetData = pow2(E, reps) * u64(offsetData) % E;
  }

  u32 getFFTSize() { return N; }
};

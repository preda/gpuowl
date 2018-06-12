#pragma once

#include "common.h"
#include "shared.h"

#include <cstring>
#include <memory>
#include <algorithm>

int extra(unsigned N, unsigned E, unsigned k) {
  assert(E % N);
  u32 step = N - (E % N);
  return u64(step) * k % N;
}

bool isBigWord(unsigned N, unsigned E, unsigned k) {
  u32 step = N - (E % N); 
  return extra(N, E, k) + step < N;
  // return extra(N, E, k) < extra(N, E, k + 1);
}

int bitlen(int N, int E, int k) { return E / N + isBigWord(N, E, k); }

u32 unbalance(int w, int nBits, int *carry) {
  assert(*carry == 0 || *carry == -1);
  w += *carry;
  *carry = 0;
  if (w < 0) {
    w += (1 << nBits);
    *carry = -1;
  }
  assert(0 <= w && w < (1 << nBits));
  return w;
}

std::vector<u32> compactBits(const vector<int> &dataVect, int E, int offset) {
  std::vector<u32> out;
  out.reserve((E - 1) / 32 + 1);

  int N = dataVect.size();
  const int *data = dataVect.data();

  int startWord = bitposToWord(E, N, offset);
  // offset * i64(N) / E;
  assert(startWord >= 0 && startWord < N);

  int startBit = offset - wordToBitpos(E, N, startWord);
  // (startWord * i64(E) + N - 1) / N;
  assert(startBit >= 0 && startBit < bitlen(N, E, startWord));
  
  int carry = 0;
  u32 outWord = 0;
  int haveBits = 0;

  for (int i = 0; i <= N; ++i) { // ! including N. Reason: go twice over startWord.
    int p = (i + startWord) % N;
    int nBits = bitlen(N, E, p);

    u32 w = 0;
    if (p == startWord) {
      if (i == 0) {
        w = unbalance(data[p], nBits, &carry);
        w >>= startBit;
        nBits -= startBit;
      } else {
        nBits = startBit;
        w = unbalance(data[p] & ((1 << startBit) - 1), startBit, &carry);
      }
    } else {
      w = unbalance(data[p], nBits, &carry);
    }

    assert(nBits > 0 || (i == N && startBit == 0));
    assert(w < (1u << nBits));
    
    while (nBits) {
      assert(haveBits < 32);
      outWord |= w << haveBits;
      if (haveBits + nBits >= 32) {
        w >>= (32 - haveBits);
        nBits -= (32 - haveBits);
        out.push_back(outWord);
        outWord = 0;
        haveBits = 0;
      } else {
        haveBits += nBits;
        nBits = 0;
      }
    }
  }

  assert(haveBits);
  out.push_back(outWord);

  for (int p = 0; carry; ++p) {
    i64 v = i64(out[p]) + carry;
    out[p] = v & 0xffffffff;
    carry = v >> 32;
  }

  assert(int(out.size()) == (E - 1) / 32 + 1);
  return out;
}

std::vector<int> expandBits(const std::vector<u32> &compactBits, int N, int E) {
  // This is similar to carry propagation.

  std::vector<int> out(N);
  int *data = out.data();
  
  int haveBits = 0;
  u64 bits = 0;

  assert(E % 32 != 0);
  auto it = compactBits.cbegin(), itEnd = compactBits.cend();
  for (int p = 0; p < N; ++p) {
    int len = bitlen(N, E, p);
    if (haveBits < len) {
      assert(it != itEnd);
      bits += u64(*it++) << haveBits;
      haveBits += 32;
    }
    assert(haveBits >= len);
    int b = bits & ((1 << len) - 1);
    bits >>= len;

    // turn the (len - 1) bit of b into sign bit.
    b = (b << (32 - len)) >> (32 - len);
    if (b < 0) { ++bits; }

    data[p] = b;
    // bits = (bits - b) >> len;
    haveBits -= len;
  }
  assert(it == itEnd);
  assert(haveBits == 32 - E % 32);
  assert(bits == 0 || bits == 1);
  data[0] += bits;
  return out;
}

#pragma once

#include "common.h"
#include "shared.h"

#include <cassert>
#include <memory>
#include <algorithm>

u32 step(u32 N, u32 E) { return N - (E % N); }
u32 extra(u32 N, u32 E, u32 k) { return u64(step(N, E)) * k % N; }
bool isBigWord(u32 N, u32 E, u32 k) { return extra(N, E, k) + step(N, E) < N; }
u32 bitlen(u32 N, u32 E, u32 k) { return E / N + isBigWord(N, E, k); }

/*
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
*/

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
  assert(startBit >= 0 && startBit < int(bitlen(N, E, startWord)));
  
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

#ifndef DUAL
#define DUAL
#endif

DUAL int lowBits(int u, int bits) { return (u << (32 - bits)) >> (32 - bits); }

struct BitBucket {
  u64 bits;
  int size;

  BitBucket() : bits(0), size(0) {}

  void put32(u32 b) {
    assert(size <= 32);
    bits += (u64(b) << size);
    size += 32;
  }
  
  int popSigned(int n) {
    assert(size >= n);
    int b = lowBits(bits, n);
    size -= n;
    bits >>= n;
    bits += (b < 0); // carry fixup.
    return b;
  }
};

vector<int> expandBits(const vector<u32> &compactBits, int N, int E) {
  assert(E % 32 != 0);

  std::vector<int> out(N);
  int *data = out.data();
  BitBucket bucket;
  
  auto it = compactBits.cbegin(), itEnd = compactBits.cend();
  for (int p = 0; p < N; ++p) {
    int len = bitlen(N, E, p);    
    if (bucket.size < len) { assert(it != itEnd); bucket.put32(*it++); }
    data[p] = bucket.popSigned(len);
  }
  assert(it == itEnd);
  assert(bucket.size == 32 - E % 32 && (bucket.bits == 0 || bucket.bits == 1));
  data[0] += bucket.bits; // carry wrap-around.
  return out;
}

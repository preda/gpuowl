// Copyright 2017 Mihai Preda.

#include "state.h"
#include "shared.h"

#include <cassert>
#include <memory>
#include <cmath>

static u32 bitlen(u32 N, u32 E, u32 k) { return E / N + isBigWord(N, E, k); }

static int lowBits(int u, int bits) { return (u << (32 - bits)) >> (32 - bits); }

static u32 unbalance(int w, int nBits, int *carry) {
  assert(*carry == 0 || *carry == -1);
  w += *carry;
  *carry = 0;
  if (w < 0) {
    w += (1 << nBits);
    *carry = -1;
  }
  if (!(0 <= w && w < (1 << nBits))) { log("w=%d, nBits=%d\n", w, nBits); }
  assert(0 <= w && w < (1 << nBits));
  return w;
}

std::vector<u32> compactBits(const vector<int> &dataVect, u32 E) {
  std::vector<u32> out;
  out.reserve((E - 1) / 32 + 1);

  u32 N = dataVect.size();
  const int *data = dataVect.data();

  int carry = 0;
  u32 outWord = 0;
  int haveBits = 0;

  for (u32 p = 0; p < N; ++p) {
    int nBits = bitlen(N, E, p);
    u32 w = unbalance(data[p], nBits, &carry);

    assert(nBits > 0);
    assert(w < (1u << nBits));

    assert(haveBits < 32);
    int topBits = 32 - haveBits;
    outWord |= w << haveBits;
    if (nBits >= topBits) {
      out.push_back(outWord);
      outWord = w >> topBits;
      haveBits = nBits - topBits;
    } else {
      haveBits += nBits;
    }
  }

  assert(haveBits);
  out.push_back(outWord);

  for (int p = 0; carry; ++p) {
    i64 v = i64(out[p]) + carry;
    out[p] = v & 0xffffffff;
    carry = v >> 32;
  }

  assert(out.size() == (E - 1) / 32 + 1);
  return out;
}

struct BitBucket {
  u64 bits;
  u32 size;

  BitBucket() : bits(0), size(0) {}

  void put32(u32 b) {
    assert(size <= 32);
    bits += (u64(b) << size);
    size += 32;
  }
  
  int popSigned(u32 n) {
    assert(size >= n);
    int b = lowBits(bits, n);
    size -= n;
    bits >>= n;
    bits += (b < 0); // carry fixup.
    return b;
  }
};

vector<int> expandBits(const vector<u32> &compactBits, u32 N, u32 E) {
  assert(E % 32 != 0);

  std::vector<int> out(N);
  int *data = out.data();
  BitBucket bucket;
  
  auto it = compactBits.cbegin(), itEnd = compactBits.cend();
  for (u32 p = 0; p < N; ++p) {
    u32 len = bitlen(N, E, p);    
    if (bucket.size < len) { assert(it != itEnd); bucket.put32(*it++); }
    data[p] = bucket.popSigned(len);
  }
  assert(it == itEnd);
  assert(bucket.size == 32 - E % 32);
  assert(bucket.bits == 0 || bucket.bits == 1);
  data[0] += bucket.bits; // carry wrap-around.
  return out;
}

u64 residueFromRaw(u32 N, u32 E, const vector<int> &words) {
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


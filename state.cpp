// Copyright 2017 Mihai Preda.

#include "state.h"
#include "shared.h"

#include <cassert>
#include <memory>
#include <cmath>
#include <iostream>

static u32 bitlen(u32 N, u32 E, u32 k) { return E / N + isBigWord(N, E, k); }

static i64 lowBits(i64 u, int bits) { return (u << (64 - bits)) >> (64 - bits); }

static u32 unbalance(i32 w, u32 nBits, i64& carry) {
  assert(carry + w == i128(carry) + w); // no overflow
  carry += w;

  u32 bits = carry & ((1u << nBits) - 1);
  carry >>= nBits;
  return bits;
}

static u32 unbalance(i32 w, u32 nBits, i32& carry) {
  assert(carry + w == i128(carry) + w); // no overflow
  carry += w;

  u32 bits = carry & ((1u << nBits) - 1);
  carry >>= nBits;
  return bits;
}

void carryTail(vector<u32> out, i64 carry) {
  for (int p = 0; carry; ++p) {
    carry += out[p];
    out[p] = u32(carry);
    carry >>= 32;
  }  
}

std::vector<u32> compact(const vector<i32>& data, u32 E, i32 mul) {
  std::vector<u32> out;
  out.reserve((E - 1) / 32 + 1);

  u32 N = data.size();
  
  i32 carry = 0;
  u32 outWord = 0;
  u32 haveBits = 0;

  for (u32 p = 0; p < N; ++p) {
    u32 nBits = bitlen(N, E, p);
    u32 w = unbalance(data[p] * mul, nBits, carry);

    assert(haveBits < 32);
    u32 topBits = 32u - haveBits;
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

  carryTail(out, carry);

  assert(out.size() == (E - 1) / 32 + 1);
  return out;
}

std::vector<u32> compactBits(const vector<i32>& data, const vector<i64>& carries, u32 E) {
  std::vector<u32> out;
  out.reserve((E - 1) / 32 + 1);

  u32 N = data.size();
  assert(N == carries.size() * 4);
  
  i64 carry = carries.back();
  u32 outWord = 0;
  u32 haveBits = 0;

  for (u32 p = 0; p < N; ++p) {
    if (p && (p % 4 == 0)) {
      // if (carries[p/4 - 1]) { cout << carry << ' ' << carries[p/4-1] << endl; }
      carry += carries[p / 4 - 1];
    }

    u32 nBits = bitlen(N, E, p);
    u32 w = unbalance(data[p], nBits, carry);

    assert(haveBits < 32);
    u32 topBits = 32u - haveBits;
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

  carryTail(out, carry);

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


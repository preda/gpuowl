// Copyright 2017 Mihai Preda.

#include "state.h"
#include "shared.h"

#include <cassert>
#include <memory>
#include <cmath>

u32 bitlen(u32 N, u32 E, u32 k) { return E / N + isBigWord(N, E, k); }

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
  
  u32 prevLen = 0;
  auto it = compactBits.cbegin(), itEnd = compactBits.cend();
  for (u32 p = 0; p < N; ++p) {
    u32 len = bitlen(N, E, p);    
    if (bucket.size < len) { assert(it != itEnd); bucket.put32(*it++); }
    data[p] = bucket.popSigned(len);
    if (p == 106) { log("data[106]=%d\n", data[p]); }
    if (p == 106 && data[p] == 4) { log("here\n"); }
    if (p > 0 && data[p-1] == -(1 << (prevLen - 1)) && data[p] > 0) {
      if (p == 106 && data[p] == 4) { log("inside\n"); }
      data[p-1] += (1 << prevLen);
      --data[p];
    }
    prevLen = len;
  }
  assert(it == itEnd);
  assert(bucket.size == 32 - E % 32);
  assert(bucket.bits == 0 || bucket.bits == 1);

  data[0] += bucket.bits; // carry wrap-around.

  if (data[N-1] == -(1 << (prevLen - 1)) && data[0] > 0) {
    data[N-1] += 1 << prevLen;
    --data[0];
  }

  for (u32 p = 0; ; ++p) {
    u32 len = bitlen(N, E, p);
    if (data[p] > (1 << (len - 1)) || (data[p] == (1 << (len - 1)) && data[p+1] < 0)) {
      data[p] -= 1 << len;
      ++data[p+1];
    } else {
      break;
    }
  }

  return out;
}

u64 residueFromRaw(u32 N, u32 E, const vector<int> &words) {
  assert(words.size() == 128);
  int carry = 0;
  // int k = N - 64;
  for (int i = 0; i < 64; ++i) {
    // u32 len = bitlen(N, E, k);
    // if ()
    carry = (words[i] + carry < 0) ? -1 : 0;
  }
  
  u64 res = 0;
  int k = 0, hasBits = 0;
  for (auto p = words.begin() + 64, end = words.end(); p < end && hasBits < 64; ++p, ++k) {
    u32 len = bitlen(N, E, k);
    int w = *p + carry;
    carry = (w < 0) ? -1 : 0;
    if (w < 0) {
      w += (1 << len);
    }
    assert(w >= 0 && w < (1 << len));
    res |= u64(w) << hasBits;
    hasBits += len;
  }
  return res;
}

u32 modM31(u64 a) {
  a = (a & 0x7fffffff) + (a >> 31);
  u32 b = (a & 0x7fffffff) + (a >> 31);
  return  (b & 0x7fffffff) + (b >> 31);
}

static u32 ROL31(u32 w, u32 shift) {
  return ((w << shift) & 0x7fffffff) + (w >> (31 - shift));
}

u32 modM31(const vector<u32>& words) {
  u64 sum = 0;
  u32 shift = 0;
  for (u32 i = 0; i < words.size(); ++i) {
    u32 w = words[i];
    sum += ROL31(w, shift);
    ++shift;
    shift = (shift >= 31) ? shift - 31 : shift;
  }
  return modM31(sum);
}

/*
u32 modM31(u32 N, u32 E, vector<i32>& words) {
  u32 backLen = bitlen(N, E, N-1);
  assert(abs(words.back()) <= (1 << (backLen - 1)));
  bool tweakedBack = false;
  if (words.back() < 0) {
    words.back() += 1 << backLen;
    words.front() -= 1;
    tweakedBack = true;
  }


  u64 sum = 0;
  u32 shift = 0;
  for (u32 i = 0; i < words.size() - 1; ++i) {
    u32 len = bitlen(N, E, i);
    i32 iw = words[i];
    assert(abs(iw) <= (1 << (len - 1)));
    bool neg = (iw < 0);
    u32 w = neg ? iw + 0x7fffffff : iw;
    sum += ROL31(w, shift);
    shift += len;
    shift = (shift >= 31) ? shift - 31 : shift;
  }

  u32 i = words.size() - 1;
  u32 len = bitlen(N, E, i);
  i32 iw = words[i];
  assert(abs(iw) <= (1 << (len - 1)));
  // bool neg = (iw < 0);
  u32 w = (iw >> 31) ? iw + (1 << len) : iw;
  sum += ROL31(w, shift) + (iw >> 31);
  // if (neg) { --sum; }
  return modM31(sum);
}
*/

/*
u32 modM31(u32 N, u32 E, const vector<i32>& words) {
  u64 sum = 0;
  for (u32 i = 0, shift = 0; i < words.size(); ++i) {
    u32 len = bitlen(N, E, i);
    i32 iw = words[i];
    assert(abs(iw) <= (1 << len));
    bool neg = (iw < 0);
    u32 w = neg ? iw + (1 << len) : iw;
    sum += ROL31(w, shift);

    shift += len;
    shift = (shift >= 31) ? shift - 31 : shift;

    const u32 MINUS1 = (1u<<31) - 2u;
    if (neg) {
      sum += ROL31(MINUS1, (i < words.size() - 1) ? shift : 0);
    }
  }
  return modM31(sum);
}
*/

u32 modM31(u32 N, u32 E, const vector<i32>& words) {
  u64 sum = 0;
  i32 carry = 0;
  for (u32 i = 0, shift = 0; i < words.size(); ++i) {
    u32 len = bitlen(N, E, i);
    i32 iw = words[i];
    assert(abs(iw) <= (1 << (len - 1)));
    iw += carry;
    carry = iw >> 31;
    u32 w = carry ? iw + (1 << len) : iw;
    sum += ROL31(w, shift);
    shift += len;
    shift = (shift >= 31) ? shift - 31 : shift;
  }
  sum += carry;
  return modM31(sum);
}














// Copyright Mihai Preda

#pragma once

#include "sha3.h"

class Sha3Hash {
  SHA3Context context;
  static const constexpr int SIZE_BITS = 256;
  
public:
  Sha3Hash() { clear(); }

  void update(const unsigned char *data, u32 size) { SHA3Update(&context, data, size); }
  
  array<u64, 4> finish() && {
    u64 *p = reinterpret_cast<u64 *>(SHA3Final(&context));
    return {p[0], p[1], p[2], p[3]};
  }

  void clear() { SHA3Init(&context, SIZE_BITS); }
};

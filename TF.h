#pragma once

#include "common.h"

class TF {

public:
  virtual ~TF() {};

  virtual u64 factor(u32 exp, int bitLo, int bitHi, u64 *outBeginK, u64 *outEndK) = 0;
};

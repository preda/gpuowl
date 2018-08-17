#pragma once

#include "common.h"

class TF {

public:
  virtual ~TF() {};

  virtual u64 findFactor(u32 exp, int bitLo, int bitHi, int nDone, int nTotal, u64 *outBeginK, u64 *outEndK) = 0;

  static bool enabled();
};

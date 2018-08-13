#pragma once

#include "TF.h"

class OpenTF : public TF {  
public:
  static unique_ptr<TF> make(Args &args) {
    return std::make_unique<OpenTF>();
  }

  OpenTF() {
  }
  
  u64 factor(u32 exp, int bitLo, int bitHi, u64 *outBeginK, u64 *outEndK) {
    return 0;
  }
};

#pragma once

#include "timeutil.h"
#include "common.h"

#include <future>
#include <string>
#include <chrono>
#include <vector>

class GCD {
  future<string> gcdFuture;
  Timer timer;
  
public:
  void start(u32 E, const vector<u32> &bits, u32 sub);
  bool isOngoing() { return gcdFuture.valid(); }
  bool isReady() { return isOngoing() && gcdFuture.wait_for(chrono::steady_clock::duration::zero()) == future_status::ready; }      
  string get();
  void wait() { gcdFuture.wait(); }
};

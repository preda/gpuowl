// Copyright Mihai Preda

#pragma once

#include "common.h"

#include <thread>

class Background {
private:
  std::thread thread;
  
public:
  ~Background() { wait(); }

  template<typename T> void run(T func) {
    wait();
    thread = std::thread(func);
  }

  void wait() {
    if (thread.joinable()) {
      log("wating for background tasks..");
      thread.join();
    }
  }
};

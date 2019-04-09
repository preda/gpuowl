// Copyright Mihai Preda

#pragma once

#include <thread>

class Background {
private:
  std::thread thread;
  
public:
  template<typename T> void run(T func) {
    wait();
    thread = std::thread(func);
  }

  void wait() {
    if (thread.joinable()) { thread.join(); }
  }
};

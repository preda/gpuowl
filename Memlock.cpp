// Copyright Mihai Preda.

#include "Memlock.h"
#include "File.h"
#include "common.h"

#include <thread>
#include <chrono>

namespace {

error_code& noThrow() {
  static error_code dummy;
  return dummy;
}

}

Memlock::Memlock(fs::path base, u32 device) : lock{base / ("memlock-"s + to_string(device))} {

  bool first = true;
  while (!fs::create_directory(lock, noThrow())) {
    if (first) {
      log("Waiting for memory lock '%s'\n", lock.string().c_str());
      first = false;
    }
    std::this_thread::sleep_for(std::chrono::seconds(15));
  }
  log("Acquired memory lock '%s'\n", lock.string().c_str());
}

Memlock::~Memlock() {
  fs::remove(lock, noThrow());
  log("Released memory lock '%s'\n", lock.string().c_str());
}

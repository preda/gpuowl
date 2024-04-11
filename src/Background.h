// Copyright (C) Mihai Preda

#pragma once

#include "log.h"
#include "typeName.h"

#include <string>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <deque>

class Background {
  unsigned maxSize;
  std::deque<std::function<void()> > tasks;
  std::jthread thread;
  std::mutex mut;
  std::condition_variable cond;
  bool stopRequested{};

  void run() {
    std::function<void()> task;

    while (true) {
      {
        std::unique_lock lock(mut);
        while (tasks.empty()) {
          if (stopRequested) {
            return;
          } else {
            cond.wait(lock);
          }
        }
        task = tasks.front();
      }

      try {
        task();
      } catch (const char *mes) {
        log("Exception \"%s\"\n", mes);
      } catch (const std::string& mes) {
        log("Exception \"%s\"\n", mes.c_str());
      } catch (const std::exception& e) {
        log("Exception %s: %s\n", typeName(e), e.what());
      }

      {
        std::unique_lock lock(mut);
        assert(!tasks.empty());
        tasks.pop_front();
        if (tasks.size() == maxSize - 1 || tasks.empty()) { cond.notify_all(); }
      }
    }
  }

public:
  Background(unsigned size = 2) :
    maxSize{size},
    thread{&Background::run, this} {
  }

  ~Background() {
    std::lock_guard lock(mut);
    stopRequested = true;
    cond.notify_all();
  }

  void waitEmpty() {
    std::unique_lock lock(mut);
    while (!tasks.empty()) { cond.wait(lock); }
  }

  template<typename T> void operator()(T task) {
    std::unique_lock lock(mut);
    while (tasks.size() >= maxSize) {
      cond.wait(lock);
    }
    tasks.push_back(task);
    cond.notify_all();
  }
};

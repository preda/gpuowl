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
#include <vector>

class Background {
  std::vector<std::function<void()> > tasks;
  std::jthread thread;
  std::mutex mut;
  std::condition_variable cond;
  bool stopRequested{};

  void run() {
    while (true) {
      decltype(tasks) localTasks;

      {
        std::unique_lock lock(mut);
        while (tasks.empty()) {
          if (stopRequested) {
            return;
          } else {
            cond.wait(lock);
          }
        }
        std::swap(tasks, localTasks);
      }

      assert(!localTasks.empty());
      for (auto it = localTasks.begin(), end = localTasks.end(); it != end; ++it) {
        try {
          (*it)();
        } catch (const char *mes) {
          log("Exception \"%s\"\n", mes);
        } catch (const std::string& mes) {
          log("Exception \"%s\"\n", mes.c_str());
        } catch (const std::exception& e) {
          log("Exception %s: %s\n", typeName(e), e.what());
        }
      }
    }
  }

public:
  Background() :
    thread{&Background::run, this}
  {

  }

  ~Background() {
    std::lock_guard lock(mut);
    stopRequested = true;
    cond.notify_all();
  }

  template<typename T> void operator()(T task) {
    std::lock_guard lock(mut);
    tasks.push_back(task);
    cond.notify_all();
  }

};

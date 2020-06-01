// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) Mihai Preda.

#pragma once

#include "Args.h"
#include "Task.h"
#include "common.h"
#include <optional>

class Worktodo {
public:
  static std::optional<Task> getTask(Args &args);
  static bool deleteTask(const Task &task);
  static void deletePRP(u32 exponent);
  
  static Task makePRP(Args &args, u32 exponent) { return Task{Task::PRP, exponent}; }
  
  static Task makePM1(Args &args, u32 exponent) {
    Task task{Task::PM1, exponent};
    task.adjustBounds(args);
    return task;
  }

  static Task makeLL(Args& args, u32 exponent) { return Task{Task::LL, exponent}; }
  static Task makeVerify(Args& args, string path) { return Task{Task::VERIFY, .verifyPath=path}; }
};

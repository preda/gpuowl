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
  
  static Task makePRP(Args &args, u32 exponent) {
    Task task{Task::PRP, exponent};
    return task;
  }

  static Task makeVerify(Args& args, string path) { return Task{.kind=Task::VERIFY, .verifyPath=path}; }
};

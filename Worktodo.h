// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "Args.h"
#include "Task.h"
#include "common.h"

class Worktodo {
public:
  static Task getTask(Args &args);
  static bool deleteTask(const Task &task);

  static Task makePRP(Args &args, u32 exponent);
  
  static Task makePM1(Args &args, u32 exponent) {
    Task task{Task::PM1, exponent};
    task.adjustBounds(args);
    return task;
  }
};

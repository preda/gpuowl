// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"

class Task;
class Args;

class Worktodo {
public:
  static Task getTask(Args &args);
  static bool deleteTask(const Task &task);

  static Task makePRP(Args &args, u32 exponent);
  static Task makePM1(Args &args, u32 exponent);
};

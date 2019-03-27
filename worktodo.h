// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

class Task;
class Args;

class Worktodo {
public:
  static Task getTask(Args &args);
  static bool deleteTask(const Task &task);
};

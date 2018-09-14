// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

class Task;

class Worktodo {
public:
  static Task getTask();
  static bool deleteTask(const Task &task);
};

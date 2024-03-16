// Copyright (C) Mihai Preda.

#pragma once

#include "common.h"
#include <optional>

class Task;
class Args;

class Worktodo {
public:
  static std::optional<Task> getTask(Args &args, i32 instance);
  static bool deleteTask(const Task &task, i32 instance);
};

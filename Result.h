// Copyright 2017 Mihai Preda.

#pragma once

#include "common.h"

#include <string>

class Task;
class Args;

struct TFResult {
  string factor;
  u64 beginK;
  u64 endK;

  bool write(const Args &args, const Task &task);
};

struct PRPResult {
  string factor;
  bool isPrime;
  u64 res64;
  u64 baseRes64;

  bool write(const Args &args, const Task &task, u32 fftSize);
};

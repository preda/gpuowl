// Copyright Mihai Preda.

#pragma once

#include "common.h"

class Task;
class Args;

struct PRPResult {
  bool isPrime;
  u64 res64;
  
  bool write(const Args &args, const Task &task, u32 fftSize);
};

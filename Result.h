// Copyright 2017 Mihai Preda.

#pragma once

#include "common.h"

#include <string>

class Task;
class Args;

struct PRPResult {
  string factor;
  bool isPrime;
  u64 res64;
  u64 baseRes64;
  u32 B2;
  
  bool write(const Args &args, const Task &task, u32 fftSize);
};

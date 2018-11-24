// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"
#include <string>
#include <memory>

class Args;
class Result;

struct Task {
  enum Kind {NONE = 0, PRP};

  Kind kind;
  u32 exponent;
  string AID;  
  string line; // the verbatim worktodo line, used in deleteTask().

  // PRP,P-1
  u32 B1;
  u32 B2;

  operator bool() { return kind != NONE; }

  bool execute(const Args &args);
};

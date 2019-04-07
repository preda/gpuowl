// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"
#include <string>

class Args;
class Result;

struct Task {
  enum Kind {NONE = 0, PRP, PM1};

  Kind kind;
  u32 exponent;
  string AID;  
  string line; // the verbatim worktodo line, used in deleteTask().

  // PM1
  u32 B1 = 0;
  u32 B2 = 0;

  operator bool() { return kind != NONE; }

  bool execute(const Args &args);

  bool writeResultPRP(const Args&, bool isPrime, u64 res64, u32 fftSize) const;
  bool writeResultPM1(const Args&, const std::string& factor, u32 fftSize) const;
};

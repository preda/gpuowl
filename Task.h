// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "Args.h"
#include "common.h"
#include <string>

class Args;
class Result;
class Background;

struct Task {
  enum Kind {PRP, PM1};

  Kind kind;
  u32 exponent;
  string AID;  // Assignment ID
  string line; // the verbatim worktodo line, used in deleteTask().

  // PM1
  u32 B1 = 0;
  u32 B2 = 0;

  void adjustBounds(Args& args);
  
  bool execute(const Args& args, Background& background);

  void writeResultPRP(const Args&, bool isPrime, u64 res64, u32 fftSize, u32 nErrors) const;
  void writeResultPM1(const Args&, const std::string& factor, u32 fftSize, bool didStage2) const;
};

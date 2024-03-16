// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "Args.h"
#include "common.h"
#include <string>

class Args;
class Result;
class Context;

struct Task {
  enum Kind {PRP, VERIFY, LL};

  Kind kind;
  u32 exponent;
  string AID;  // Assignment ID
  string line; // the verbatim worktodo line, used in deleteTask().

  string verifyPath; // For Verify
  void execute(Context& context, const Args& args);

  void writeResultPRP(const Args&, bool isPrime, u64 res64, u32 fftSize, u32 nErrors, const fs::path& proofPath) const;
  void writeResultLL();
};

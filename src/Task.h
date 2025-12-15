// Copyright (C) Mihai Preda

#pragma once

#include "Args.h"
#include "common.h"
#include "GpuCommon.h"
#include "FFTConfig.h"

#include <string>

class Args;
class Result;
class Context;
class Queue;
class TrigBufCache;

class Task {
public:
  enum Kind {PRP, VERIFY, LL, CERT};

  Kind kind;
  u32 exponent;
  string AID;  // Assignment ID
  string line; // the verbatim worktodo line, used in deleteTask().
  u32 squarings;  // For CERTs

  string verifyPath; // For Verify
  void execute(GpuCommon shared, Queue* q, u32 instance);

  void writeResultPRP(FFTConfig fft, const Args&, u32 instance, bool isPrime, u64 res64, const std::string& res2048, u32 nErrors, const fs::path& proofPath) const;
  void writeResultLL(FFTConfig fft, const Args&, u32 instance, bool isPrime, u64 res64) const;
  void writeResultCERT(FFTConfig fft, const Args&, u32 instance, array <u64, 4> hash, u32 squarings) const;
};

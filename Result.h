// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"

#include <string>

class Task;
class Args;

class Result {
protected:
  string factor;

public:
  Result(const string &factor) : factor(factor) {
  }
  
  virtual ~Result();
  virtual bool write(const Args &args, const Task &task) = 0;
};

class TFResult : public Result {
  u64 beginK, endK;

public:
  TFResult(const string &factor, u64 beginK, u64 endK) :
    Result(factor),
    beginK(beginK),
    endK(endK) {
  }

  bool write(const Args &args, const Task &task) override;
};

/*
class PFResult : public Result {
public:
  PFResult(const string &factor) :
    Result(factor) {
  }

  bool write(const Args &args, const Task &task) override;
};
*/

class PRPResult : public Result {
  bool isPrime;
  u32 B1;
  u64 res64;
  u64 baseRes64;

public:
  PRPResult(const string &factor, bool isPrime, u32 B1, u64 res64, u64 baseRes64):
    Result(factor),
    isPrime(isPrime),
    B1(B1),
    res64(res64),
    baseRes64(baseRes64) {
  }

  bool write(const Args &args, const Task &task) override;
};

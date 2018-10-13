// GpuOwL, a Mersenne primality tester. Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"
#include <string>

class Args {
private:
  
public:
  enum {CARRY_AUTO = 0, CARRY_SHORT, CARRY_LONG};

  u32 B1;
  u32 B2;
  std::string clArgs;
  std::string user, cpu;
  std::string dump;
  int device;
  bool timeKernels;
  bool listFFT;
  int carry;
  u32 blockSize;
  int fftSize;
  int tfDelta;
  bool enableTF;
  bool usePrecompiled;
  
  Args() :
    B1(0),
    B2(0),
    device(-1),
    timeKernels(false),
    listFFT(false),
    carry(CARRY_AUTO),
    blockSize(400),
    fftSize(0),
    tfDelta(0),
    enableTF(false),
    usePrecompiled(false)
  { }

  // return false to stop.
  bool parse(int argc, char **argv);
};

// GpuOwL, a Mersenne primality tester. Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"
#include <string>

class Args {
private:
  
public:
  enum {CARRY_AUTO = 0, CARRY_SHORT, CARRY_LONG};

  u32 B1;
  std::string clArgs;
  std::string user, cpu;
  std::string dump;
  int device;
  bool timeKernels;
  bool listFFT;
  int carry;
  int blockSize;
  int fftSize;
  int tfDelta;
  bool enableTF;
  bool usePrecompiled;
  string ksetFile;
  
  Args() :
    B1(0),
    device(-1),
    timeKernels(false),
    listFFT(false),
    carry(CARRY_AUTO),
    blockSize(1000),
    fftSize(0),
    tfDelta(0),
    enableTF(false),
    usePrecompiled(false),
    ksetFile("kset.txt")
  { }

  // return false to stop.
  bool parse(int argc, char **argv);
};

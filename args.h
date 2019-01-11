// GpuOwL, a Mersenne primality tester. Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"
#include "clwrap.h"

#include <string>
#include <vector>

class Args {
private:
  
public:
  enum {CARRY_AUTO = 0, CARRY_SHORT, CARRY_LONG};

  std::string clArgs;
  std::string user, cpu;
  std::string dump;
  std::vector<u32> devices;
  bool timeKernels;
  int carry;
  u32 blockSize;
  int fftSize;
  int tfDelta;
  bool enableTF;
  bool usePrecompiled;
  
  Args() :
    timeKernels(false),
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

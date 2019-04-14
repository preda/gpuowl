// Copyright Mihai Preda.

#pragma once

#include "common.h"

#include <string>

class Args {
public:
  enum {CARRY_AUTO = 0, CARRY_SHORT, CARRY_LONG};

  void parse(string line);
  void parse(int argc, char **argv);
  
  string user;
  string cpu;
  string dump;
  string dir;
  
  int device = -1;
  bool timeKernels = false;
  int carry = CARRY_AUTO;
  u32 blockSize = 400;
  int fftSize = 0;
  bool enableTF = false;
  u32 B1 = 500000;
  u32 B2 = 0;
  u32 B2_B1_ratio = 30;

  u32 prpExp = 0;
  u32 pm1Exp = 0;
};

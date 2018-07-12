// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#pragma once

#include "common.h"

#include <cstdio>
#include <cstring>
#include <cassert>

// It seems that 308M is impossible to do with a 16M FFT, so that's the upper limit.
const int EXP_MIN = 1024, EXP_MAX = 370000000;

int parseLine(const char *line, char *outAID) {
  int exp = 0;
  outAID[0] = 0;
  
  if (false
      || (sscanf(line, "Test=%32[0-9a-fA-F],%d", outAID, &exp) == 2)
      || (sscanf(line, "DoubleCheck=%32[0-9a-fA-F],%d", outAID, &exp) == 2)
      || (sscanf(line, "PRP=%32[0-9a-fA-F],%*d,%*d,%d", outAID, &exp) == 2)
      || (sscanf(line, "Test=%d", &exp) == 1)
      || (sscanf(line, "%d", &exp) == 1)) {
    return exp;
  }
  
  outAID[0] = 0;
  return 0;
}

int worktodoReadExponent(char *AID) {
  auto fi{open("worktodo.txt", "r")};
  if (!fi) { return 0; }

  while (true) {
    char line[256];
    if (!fgets(line, sizeof(line), fi.get())) { break; }
    
    if (int exp = parseLine(line, AID)) {
      if (exp >= EXP_MIN && exp <= EXP_MAX) {
        return exp;
      } else {
        log("Exponent %d skipped: must be between %d and %d\n", exp, EXP_MIN, EXP_MAX);
      }
    } else {
      int n = strlen(line);
      if (n >= 2 && line[n - 2] == '\n') { line[n - 2] = 0; }
      if (n >= 1 && line[n - 1] == '\n') { line[n - 1] = 0; }
      log("worktodo.txt: \"%s\" ignored\n", line);
    }
  }
  return 0;
}

bool worktodoDelete(int E) {
  bool lineDeleted = false;

  {
    auto fi{open("worktodo.txt", "rb")};
    auto fo{open("worktodo-tmp.tmp", "wb")};
    if (!(fi && fo)) { return false; }

    char line[512];
    char AID[64];
    while (fgets(line, sizeof(line), fi.get())) {
      if (!lineDeleted && (parseLine(line, AID) == E)) {
        lineDeleted = true;
      } else {
        fputs(line, fo.get());
      }
    }
  }

  return lineDeleted && (rename("worktodo.txt", "worktodo.bak") == 0) && (rename("worktodo-tmp.tmp", "worktodo.txt") == 0);
}

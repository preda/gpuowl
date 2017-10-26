// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#pragma once

#include "common.h"

#include <cstdio>
#include <cstring>
#include <cassert>

const int EXP_MIN = 2000000, EXP_MAX = 155000000;

int parseLine(const char *line, char *outAID) {
  char kind[32];
  int exp;
  if (sscanf(line, "%11[^=]=%32[0-9a-fA-F],%d,%*d,%*d", kind, outAID, &exp) == 3
      && (!strcmp(kind, "Test") || !strcmp(kind, "DoubleCheck"))) {
    return exp;
  }
  outAID[0] = 0;
  return (sscanf(line, "Test=%d", &exp) == 1) ? exp : 0;
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
      log("worktodo.txt line '%s' skipped\n", line);
    }
  }
  return 0;
}

bool worktodoGetLinePos(int E, int *begin, int *end) {
  auto fi{open("worktodo.txt", "r")};
  if (!fi) { return false; }

  i64 p1 = 0;
  while (true) {
    char line[256];
    if (!fgets(line, sizeof(line), fi.get())) { return false; }
    i64 p2 = ftell(fi.get());
    char AID[64];
    if (parseLine(line, AID) == E) {      
      *begin = p1;
      *end = p2;
      return true;
    }
    p1 = p2;
  }
}

bool worktodoDelete(int begin, int end) {
  assert(begin >= 0 && end > begin);

  int n = 0;
  char buf[64 * 1024];

  if (auto fi = open("worktodo.txt", "r")) {
    n = fread(buf, 1, sizeof(buf), fi.get());    
  } else {
    return true;
  }
  
  if (n == sizeof(buf) || end > n) { return false; }
  memmove(buf + begin, buf + end, n - end);

  if (auto fo{open("worktodo-tmp.tmp", "w")}) {
    int newSize = begin + n - end;
    if (newSize && (fwrite(buf, newSize, 1, fo.get()) != 1)) { return false; }    
  } else {
    return false;
  }

  remove("worktodo.bak");
  return (rename("worktodo.txt", "worktodo.bak") == 0) && (rename("worktodo-tmp.tmp", "worktodo.txt") == 0);
}

bool worktodoDelete(int E) {
  int lineBegin, lineEnd;
  return worktodoGetLinePos(E, &lineBegin, &lineEnd) && worktodoDelete(lineBegin, lineEnd);
}

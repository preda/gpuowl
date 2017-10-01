// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#pragma once

#include "common.h"

#include <cstdio>
#include <cstring>
#include <cassert>

FILE *open(const char *name, const char *mode);

const int EXP_MIN = 4000000, EXP_MAX = 155000000;

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
  FILE *fi = open("worktodo.txt", "r");
  if (!fi) { return 0; }

  char line[256];
  while (true) {
    if (fscanf(fi, "%255s\n", line) < 1) { break; }
    if (int exp = parseLine(line, AID)) {
      if (exp >= EXP_MIN && exp <= EXP_MAX) {
        fclose(fi);
        return exp;
      } else {
        log("Exponent %d skipped: must be between %d and %d\n", exp, EXP_MIN, EXP_MAX);
      }
    } else {
      log("worktodo.txt line '%s' skipped\n", line);
    }
  }
  fclose(fi);
  return 0;
}

bool worktodoGetLinePos(int E, int *begin, int *end) {
  FILE *fi = open("worktodo.txt", "r");
  if (!fi) { return false; }

  char line[256];
  char AID[64];
  i64 p1 = 0;
  while (true) {
    if (fscanf(fi, "%255s\n", line) < 1) { break; }
    i64 p2 = ftell(fi);
    if (parseLine(line, AID) == E) {      
      *begin = p1;
      *end = p2;
      fclose(fi);
      return true;
    }
    p1 = p2;
  }
  fclose(fi);
  return false;
}

bool worktodoDelete(int begin, int end) {
  assert(begin >= 0 && end > begin);
  FILE *fi = open("worktodo.txt", "r");
  if (!fi) { return true; }
  char buf[64 * 1024];
  int n = fread(buf, 1, sizeof(buf), fi);
  fclose(fi);
  if (n == sizeof(buf) || end > n) { return false; }
  memmove(buf + begin, buf + end, n - end);
  
  FILE *fo = open("worktodo-tmp.tmp", "w");
  if (!fo) { return false; }
  int newSize = begin + n - end;
  bool ok = (newSize == 0) || (fwrite(buf, newSize, 1, fo) == 1);
  fclose(fo);
  if (!ok) { return false; }
  remove("worktodo.bak");
  return (rename("worktodo.txt", "worktodo.bak") == 0) && (rename("worktodo-tmp.tmp", "worktodo.txt") == 0);
}

bool worktodoDelete(int E) {
  int lineBegin, lineEnd;
  return worktodoGetLinePos(E, &lineBegin, &lineEnd) && worktodoDelete(lineBegin, lineEnd);
}

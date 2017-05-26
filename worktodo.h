// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#pragma once

#include "common.h"

#include <cstdio>
#include <cstring>
#include <cassert>

FILE *open(const char *name, const char *mode);

const int EXP_MIN = 50000000, EXP_MAX = 78000000;

int worktodoReadExponent(char *AID) {
  FILE *fi = open("worktodo.txt", "r");
  if (!fi) { return 0; }

  char line[256];
  char kind[32];
  int exp;
  int ret = 0;
  *AID = 0;
  while (true) {
    if (fscanf(fi, "%255s\n", line) < 1) { break; }
    if (((sscanf(line, "%11[^=]=%32[0-9a-fA-F],%d,%*d,%*d", kind, AID, &exp) == 3
          && (!strcmp(kind, "Test") || !strcmp(kind, "DoubleCheck")))
         || sscanf(line, "Test=%d", &exp) == 1)
        && exp >= EXP_MIN && exp <= EXP_MAX) {
      ret = exp;
      break;
    } else {
      log("worktodo.txt line '%s' skipped\n", line);
    }
  }
  fclose(fi);
  return ret;
}

bool worktodoGetLinePos(int E, int *begin, int *end) {
  FILE *fi = open("worktodo.txt", "r");
  if (!fi) { return false; }

  char line[256];
  char kind[32];
  int exp;
  bool ret = false;
  i64 p1 = 0;
  while (true) {
    if (fscanf(fi, "%255s\n", line) < 1) { break; }
    i64 p2 = ftell(fi);
    if (sscanf(line, "%11[^=]=%*32[0-9a-fA-F],%d,%*d,%*d", kind, &exp) == 2 &&
        (!strcmp(kind, "Test") || !strcmp(kind, "DoubleCheck")) &&
        exp == E) {
      *begin = p1;
      *end = p2;
      ret = true;
      break;
    }
    p1 = p2;
  }
  fclose(fi);
  return ret;
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
  return ok &&
    (rename("worktodo.txt",     "worktodo.bak") == 0) &&
    (rename("worktodo-tmp.tmp", "worktodo.txt") == 0);
}

bool worktodoDelete(int E) {
  int lineBegin, lineEnd;
  return worktodoGetLinePos(E, &lineBegin, &lineEnd) && worktodoDelete(lineBegin, lineEnd);
}

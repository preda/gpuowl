// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#pragma once

#include "common.h"

#include <cstdio>
#include <cstring>
#include <cassert>

struct Task {
  enum Kind {NONE = 0, PRP, TF, LL};

  Kind kind;
  u32 exponent;
  string AID;  
  string line; // the verbatim worktodo line, used in deleteTask().
  int bitLo;   // Some PRP tasks contain "factored up to" bitlevel.
  
  // TF only
  int bitHi;
};

class Worktodo {
public:
  static Task getTask() {
    if (auto fi{open("worktodo.txt", "r")}) {
      char line[512];
      while (fgets(line, sizeof(line), fi.get())) {
        u32 exp = 0;
        char outAID[64] = {0};
        int bitLo = 0, bitHi = 0;
        if (false
            || (sscanf(line, "%u", &exp) == 1)          
            || (sscanf(line, "Factor=%u,%d,%d", &exp, &bitLo, &bitHi) == 3)
            || (sscanf(line, "PRP=%32[0-9a-fA-F],%*d,%*d,%u,%*d,%d", outAID, &exp, &bitLo) == 3)
            || (sscanf(line, "Factor=%32[0-9a-fA-F],%u,%d,%d", outAID, &exp, &bitLo, &bitHi) == 4)
            ) {
          return Task{bitHi ? Task::TF : Task::PRP, exp, outAID, line, bitLo, bitHi};
        }

        int n = strlen(line);
        if (n >= 2 && line[n - 2] == '\n') { line[n - 2] = 0; }
        if (n >= 1 && line[n - 1] == '\n') { line[n - 1] = 0; }
        log("worktodo.txt: \"%s\" ignored\n", line);
      }
    }
    return Task{Task::NONE};
  }

  static bool deleteTask(const Task &task) {
    bool lineDeleted = false;

    {
      auto fi{open("worktodo.txt", "rb")};
      auto fo{open("worktodo-tmp.tmp", "wb")};
      if (!(fi && fo)) { return false; }
      
      char line[512];
      while (fgets(line, sizeof(line), fi.get())) {
        if (!lineDeleted && !strcmp(line, task.line.c_str())) {
          lineDeleted = true;
        } else {
          fputs(line, fo.get());
        }
      }
    }

    return lineDeleted && (rename("worktodo.txt", "worktodo.bak") == 0) && (rename("worktodo-tmp.tmp", "worktodo.txt") == 0);
  }
};

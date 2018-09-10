// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#pragma once

#include "common.h"
#include "TF.h"

#include <cstdio>
#include <cstring>
#include <cassert>

struct Task {
  // static Task makePM1(u32 exp, u32 B1, const string &aid, const string &iniLine, 
  
  enum Kind {NONE = 0, PRP, TF, LL, PM1};

  Kind kind;
  u32 exponent;
  string AID;  
  string line; // the verbatim worktodo line, used in deleteTask().
  int bitLo;   // Some PRP tasks contain "factored up to" bitlevel.
  
  // TF only
  int bitHi;

  // PM1 only
  u32 B1;
};

class Worktodo {
public:
  static Task getTask() {
    if (auto fi{open("worktodo.txt", "rb")}) {
      char line[512];
      while (fgets(line, sizeof(line), fi.get())) {
        u32 exp = 0;
        char outAID[64] = {0};
        int bitLo = 0, bitHi = 0;

        if (sscanf(line, "%u,%d", &exp, &bitLo) == 2 ||
            sscanf(line, "%u", &exp) == 1 ||
            sscanf(line, "PRP=N/A,1,2,%u,-1,%d", &exp, &bitLo) == 2 ||
            sscanf(line, "PRP=%32[0-9a-fA-F],1,2,%u,-1,%d", outAID, &exp, &bitLo) == 3) {
          return Task{Task::PRP, exp, outAID, line, bitLo, bitHi};
        }

        outAID[0] = 0;
        if (TF::enabled() &&
            (sscanf(line, "Factor=%u,%d,%d", &exp, &bitLo, &bitHi) == 3 ||
             sscanf(line, "Factor=N/A,%u,%d,%d", &exp, &bitLo, &bitHi) == 3 ||
             sscanf(line, "Factor=%32[0-9a-fA-F],%u,%d,%d", outAID, &exp, &bitLo, &bitHi) == 4)) {
          return Task{Task::TF, exp, outAID, line, bitLo, bitHi};
        }

        outAID[0] = 0;
        u32 B1 = 0;
        if (sscanf(line, "PFactor=N/A,1,2,%u,-1", &exp) == 1 ||
            sscanf(line, "PFactor=B1:%u,%u", &B1, &exp) == 2 ||
            sscanf(line, "PFactor=B1:%u,%32[0-9a-fA-F],1,2,%u,-1", &B1, outAID, &exp) == 3 ||
            sscanf(line, "PFactor=%32[0-9a-fA-F],1,2,%u,-1", outAID, &exp) == 2) {
          return Task{Task::PM1, exp, outAID, line, 0, 0, B1};
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

    if (!lineDeleted) {
      log("worktodo.txt: could not find the line \"%s\" to delete\n", task.line.c_str());
      return false;
    }
    remove("worktodo.bak");
    rename("worktodo.txt", "worktodo.bak");
    rename("worktodo-tmp.tmp", "worktodo.txt");
    return true;
  }
};

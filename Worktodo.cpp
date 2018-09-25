#include "worktodo.h"

#include "Task.h"
#include "file.h"
#include "common.h"

#include <cassert>
#include <string>
#include <cstring>

Task Worktodo::getTask() {
  if (auto fi{openRead("worktodo.txt", true)}) {
    char line[512];
    while (fgets(line, sizeof(line), fi.get())) {
      u32 exp = 0;
      char outAID[64] = {0};
      u32 bitLo = 0, bitHi = 0;

      // temporarilly PFactor lines are parsed as PRP-1.
      
      if (sscanf(line, "%u,%d", &exp, &bitLo) == 2 ||
          sscanf(line, "%u", &exp) == 1 ||
          sscanf(line, "PRP=N/A,1,2,%u,-1,%u", &exp, &bitLo) == 2 ||
          sscanf(line, "PFactor=N/A,1,2,%u,-1,%u", &exp, &bitLo) == 2 ||
          sscanf(line, "PRP=%32[0-9a-fA-F],1,2,%u,-1,%u", outAID, &exp, &bitLo) == 3 ||
          sscanf(line, "PFactor=%32[0-9a-fA-F],1,2,%u,-1,%u", outAID, &exp, &bitLo) == 3) {
        return Task{Task::PRP, exp, outAID, line, bitLo, bitHi};
      }

      outAID[0] = 0;
      if (sscanf(line, "Factor=%u,%d,%d", &exp, &bitLo, &bitHi) == 3 ||
          sscanf(line, "Factor=N/A,%u,%d,%d", &exp, &bitLo, &bitHi) == 3 ||
          sscanf(line, "Factor=%32[0-9a-fA-F],%u,%u,%u", outAID, &exp, &bitLo, &bitHi) == 4) {
        return Task{Task::TF, exp, outAID, line, bitLo, bitHi};
      }
      
      int n = strlen(line);
      if (n >= 2 && line[n - 2] == '\n') { line[n - 2] = 0; }
      if (n >= 1 && line[n - 1] == '\n') { line[n - 1] = 0; }
      log("worktodo.txt: \"%s\" ignored\n", line);
    }
  }
  return Task{Task::NONE};
}

bool Worktodo::deleteTask(const Task &task) {
  // Some tasks don't originate in worktodo.txt and thus don't need deleting.
  if (task.line.empty()) { return true; }

  bool lineDeleted = false;
  {
    auto fi{openRead("worktodo.txt", true)};
    auto fo{openWrite("worktodo-tmp.tmp")};
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

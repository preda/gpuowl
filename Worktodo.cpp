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
      u32 B1 = 0, B2 = 0;
      u32 bitLo = 0;
      int pos = 0;
      char *tail = line;
      if (sscanf(line, "B1=%u,B2=%u;%n", &B1, &B2, &pos) == 2 ||
          sscanf(line, "B1=%u;%n", &B1, &pos) == 1) {
        tail = line + pos;
      }

      if (sscanf(tail, "%u,%d", &exp, &bitLo) == 2 ||
          sscanf(tail, "%u", &exp) == 1 ||
          sscanf(tail, "PRP=N/A,1,2,%u,-1,%u", &exp, &bitLo) == 2 ||
          sscanf(tail, "PFactor=N/A,1,2,%u,-1,%u", &exp, &bitLo) == 2 ||
          sscanf(tail, "PRP=%32[0-9a-fA-F],1,2,%u,-1,%u", outAID, &exp, &bitLo) == 3 ||
          sscanf(tail, "PFactor=%32[0-9a-fA-F],1,2,%u,-1,%u", outAID, &exp, &bitLo) == 3) {
        return Task{Task::PRP, exp, outAID, line, B1, B2};
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

// GpuOwL, a Mersenne primality tester. Copyright (C) Mihai Preda.

#include "Worktodo.h"

#include "Task.h"
#include "File.h"
#include "common.h"
#include "Args.h"

#include <cassert>
#include <cstring>
#include <string>
#include <optional>

namespace {
std::string rstripNewline(std::string s) {
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) { s.pop_back(); }
  return s;
}

std::optional<Task> parseLine(const std::string& sline) {
  const char* line = sline.c_str();
  u32 exp = 0;
  char outAID[64] = {0};
  u32 bitLo = 0;
  int pos = 0;

  if (sscanf(line, "%u,%u", &exp, &bitLo) == 2 ||
      sscanf(line, "%u", &exp) == 1 ||
      sscanf(line, "PRP=N/A,1,2,%u,-1,%u", &exp, &bitLo) == 2 ||
      sscanf(line, "PRP=%32[0-9a-fA-F],1,2,%u,-1,%u", outAID, &exp, &bitLo) == 3) {
    return {{Task::PRP, exp, outAID, line}};
  }
  outAID[0] = 0;
      
  u32 B1 = 0, B2 = 0;
  const char* tail = line;
  
  if (sscanf(line, "B1=%u,B2=%u;%n", &B1, &B2, &pos) == 2 ||
      sscanf(line, "B1=%u;%n", &B1, &pos) == 1) {
    tail = line + pos;
  }
      
  if (sscanf(tail, "PFactor=N/A,1,2,%u,-1,%u", &exp, &bitLo) == 2 ||
      sscanf(tail, "PFactor=%32[0-9a-fA-F],1,2,%u,-1,%u", outAID, &exp, &bitLo) == 3) {
    return {{Task::PM1, exp, outAID, line, B1, B2}};
  }
  outAID[0] = 0;
  if (sscanf(tail, "PFactor=%u", &exp) == 1) { return {{Task::PM1, exp, "", line, B1, B2}}; }

  log("worktodo.txt ignored: \"%s\"\n", rstripNewline(line).c_str());
  return std::nullopt;
}
}

std::optional<Task> Worktodo::getTask(Args &args) {
  auto fi = File::openRead("worktodo.txt", true);
  char line[512];
  while (fgets(line, sizeof(line), fi.get())) {
    if (auto maybeTask = parseLine(line)) {
      maybeTask->adjustBounds(args);
      return maybeTask;
    }
  }
  return std::nullopt;
}

bool Worktodo::deleteTask(const Task &task) {
  // Some tasks don't originate in worktodo.txt and thus don't need deleting.
  if (task.line.empty()) { return true; }

  bool lineDeleted = false;
  {
    auto fi{File::openRead("worktodo.txt", true)};
    auto fo{File::openWrite("worktodo-tmp.tmp")};
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

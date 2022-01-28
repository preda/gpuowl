// GpuOwL, a Mersenne primality tester. Copyright (C) Mihai Preda.

#include "Worktodo.h"

#include "Task.h"
#include "File.h"
#include "common.h"
#include "Args.h"

#include <cassert>
#include <string>
#include <optional>

namespace {

std::optional<Task> parse(const std::string& line) {
  u32 exp = 0;

  u32 bitLo = 0;
  int pos = 0;
  u32 wantsPm1 = 0;
  u32 B1 = 0, B2 = 0;

  string tail = line;
  
  if (sscanf(tail.c_str(), "B1=%u,B2=%u;%n", &B1, &B2, &pos) == 2
      || sscanf(tail.c_str(), "B1=%u;%n", &B1, &pos) == 1
      || sscanf(tail.c_str(), "B2=%u;%n", &B2, &pos) == 1) {
    tail = tail.substr(pos);
  }

  char kindStr[32] = {0};
  if(sscanf(tail.c_str(), "%11[a-zA-Z]=%n", kindStr, &pos) == 1) {
    string kind = kindStr;
    tail = tail.substr(pos);
    if (kind == "PRP" or kind == "PRPDC") {
      if (tail.find('"') != string::npos) {
        log("GpuOwl does not support PRP-CF!\n");
      } else {
        char AIDStr[64] = {0};
        if (sscanf(tail.c_str(), "%32[0-9a-fA-F],1,2,%u,-1,%u,%u", AIDStr, &exp, &bitLo, &wantsPm1) == 4
            || (AIDStr[0]=0, sscanf(tail.c_str(), "N/A,1,2,%u,-1,%u,%u", &exp, &bitLo, &wantsPm1) == 3)
            || (AIDStr[0]=0, sscanf(tail.c_str(), "1,2,%u,-1,%u,%u", &exp, &bitLo, &wantsPm1) == 3)
            || sscanf(tail.c_str(), "%32[0-9a-fA-F],%u,%u,%u", AIDStr, &exp, &bitLo, &wantsPm1) == 4
            || (AIDStr[0]=0, sscanf(tail.c_str(), "N/A,%u,%u,%u", &exp, &bitLo, &wantsPm1) == 3)
            || (AIDStr[0]=0, sscanf(tail.c_str(), "%u,%u,%u", &exp, &bitLo, &wantsPm1) == 3)
            || (AIDStr[0]=0, sscanf(tail.c_str(), "%u", &exp)) == 1) {
          string AID = AIDStr;
          if (AID == "N/A" || AID == "0") { AID = ""; }
          return {{Task::PRP, exp, AID, line, B1, B2, bitLo, wantsPm1}};
        }
      }
    }
  }
  log("worktodo.txt line ignored: \"%s\"\n", rstripNewline(line).c_str());
  return std::nullopt;
}

fs::path operator+(fs::path p, const std::string& tail) {
  p += tail;
  return p;
}

bool deleteLine(const fs::path& fileName, const std::string& targetLine) {
  assert(!targetLine.empty());
  bool lineDeleted = false;
  {
    auto fo{File::openWrite(fileName + "-tmp")};
    for (const string& line : File::openReadThrow(fileName)) {
      if (!lineDeleted && line == targetLine) {
        lineDeleted = true;
      } else {
        fo.write(line);
      }
    }
  }

  if (!lineDeleted) {
    log("'%s': could not find the line '%s' to delete\n", fileName.string().c_str(), targetLine.c_str());
    return false;
  }
  fs::remove(fileName + "-bak");
  fs::rename(fileName, fileName + "-bak");
  fs::rename(fileName + "-tmp", fileName);  
  return true;
}

std::optional<Task> firstGoodTask(const fs::path& fileName) {
  for (const string& line : File::openRead(fileName)) {
    if (optional<Task> maybeTask = parse(line)) { return maybeTask; }
  }
  return nullopt;
}

}

std::optional<Task> Worktodo::getTask(Args &args) {
  string worktodoTxt = "worktodo.txt";
  
 again:
  // Try to get a task from the local worktodo.txt
  if (optional<Task> task = firstGoodTask(worktodoTxt)) {    
    task->adjustBounds(args);
    return task;
  }
  
  if (!args.masterDir.empty()) {
    fs::path globalWorktodo = args.masterDir / worktodoTxt;
    if (optional<Task> task = firstGoodTask(globalWorktodo)) {
      File::append(worktodoTxt, task->line);
      deleteLine(globalWorktodo, task->line);
      goto again;
    }
  }
  
  return std::nullopt;
}

bool Worktodo::deleteTask(const Task &task) {
  // Some tasks don't originate in worktodo.txt and thus don't need deleting.
  if (task.line.empty()) { return true; }
  return deleteLine("worktodo.txt", task.line);
}

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

  log("worktodo.txt line ignored: \"%s\"\n", rstripNewline(line).c_str());
  return std::nullopt;
}

std::string findGoodLine(const std::string& fileName) {
  if (auto fi = File::openRead(fileName)) {
    char line[512];
    while (fgets(line, sizeof(line), fi.get())) {
      if (parseLine(line)) { return line; }
    }
  }
  return {};
}

void remove(const std::string& s) { ::remove(s.c_str()); }
void rename(const std::string& a, const std::string& b) { ::rename(a.c_str(), b.c_str()); }

bool deleteLine(const std::string& fileName, const std::string& targetLine) {
  assert(!targetLine.empty());
  bool lineDeleted = false;
  {
    auto fi{File::openRead(fileName, true)};
    auto fo{File::openWrite(fileName + "-tmp")};
    for (auto line = fi.readLine(); !line.empty(); line = fi.readLine()) {
      if (!lineDeleted && line == targetLine) {
        lineDeleted = true;
      } else {
        fo.write(line);
      }
    }
  }

  if (!lineDeleted) {
    log("'%s': could not find the line '%s' to delete\n", fileName.c_str(), targetLine.c_str());
    return false;
  }
  remove(fileName + "-bak");
  rename(fileName, fileName + "-bak");
  rename(fileName + "-tmp", fileName);  
  return true;
}

}

std::optional<Task> Worktodo::getTask(Args &args) {
  for (int pass = 0; pass < 2; ++pass) {
    std::string localWorktodo = "worktodo.txt";
    
    // Try to get a task from the local worktodo.txt
    std::string line = findGoodLine(localWorktodo);
    if (!line.empty()) {    
      Task task = parseLine(line).value();
      task.adjustBounds(args);
      return task;
    }

    // First time, try to move a task from master worktodo.txt to local
    if (pass == 0 && !args.masterDir.empty()) {
      std::string masterWorktodo = args.masterDir + '/' + "worktodo.txt";
      line = findGoodLine(masterWorktodo);
      if (!line.empty()) {
        File::append(localWorktodo, line);
        deleteLine(masterWorktodo, line);
      }
    }
  }
  
  return std::nullopt;
}

bool Worktodo::deleteTask(const Task &task) {
  // Some tasks don't originate in worktodo.txt and thus don't need deleting.
  if (task.line.empty()) { return true; }
  return deleteLine("worktodo.txt", task.line);
}

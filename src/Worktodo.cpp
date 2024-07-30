// Copyright (C) Mihai Preda.

#include "Worktodo.h"

#include "CycleFile.h"
#include "Task.h"
#include "File.h"
#include "common.h"
#include "Args.h"

#include <cassert>
#include <string>
#include <optional>
#include <mutex>
#include <charconv>

namespace {

bool isHex(const string& s) {
  u32 dummy{};
  const char *end = s.c_str() + s.size();
  auto[ptr, ec] = std::from_chars(s.c_str(), end, dummy, 16);
  return (ptr == end);
}

// Examples:
// PRP=FEEE9DCD59A0855711265C1165C4C693,1,2,124647911,-1,77,0
// DoubleCheck=E0F583710728343C61643028FBDBA0FB,70198703,75,1
std::optional<Task> parse(const std::string& line) {
  if (line.empty() || line[0] == '#') { return {}; }

  vector<string> topParts = split(line, '=');

  bool isPRP = false;
  bool isLL = false;

  if (topParts.size() == 2) {
    string kind = topParts.front();
    if (kind == "PRP" || kind == "PRPDC") {
      isPRP = true;
    } else if (kind == "Test" || kind == "DoubleCheck") {
      isLL = true;
    }
  }

  if (isPRP || isLL) {
    vector<string> parts = split(topParts.back(), ',');
    if (!parts.empty() && (parts.front() == "N/A" || parts.front().empty())) {
      parts.erase(parts.begin()); // skip empty AID
    }

    string AID;
    if (!parts.empty() && parts.front().size() == 32 && isHex(parts.front())) {
      AID = parts.front();
      parts.erase(parts.begin());
    }

    string s = (parts.size() >= 4 && parts[0] == "1" && parts[1] == "2" && parts[3] == "-1") ? parts[2]
      : (!parts.empty() ? parts[0] : "");

    const char *end = s.c_str() + s.size();
    u64 exp{};
    auto [ptr, _] = from_chars(s.c_str(), end, exp, 10);
    if (ptr != end) { exp = 0; }
    if (exp > 1000) { return {{isPRP ? Task::PRP : Task::LL, u32(exp), AID, line}}; }
  }
  log("worktodo.txt line ignored: \"%s\"\n", rstripNewline(line).c_str());
  return {};
}

bool deleteLine(const fs::path& fileName, const std::string& targetLine) {
  assert(!targetLine.empty());
  bool lineDeleted = false;

  CycleFile fo{fileName};
  for (const string& line : File::openReadThrow(fileName)) {
    // log("line '%s'\n", line.c_str());
    if (!lineDeleted && line == targetLine) {
      lineDeleted = true;
    } else {
      fo->write(line);
    }
  }

  if (!lineDeleted) {
    log("'%s': did not find the line '%s' to delete\n", fileName.string().c_str(), targetLine.c_str());
    fo.reset();
  }

  return lineDeleted;
}

// Among the valid tasks from fileName, return the "best" which means with the smallest exponent
static std::optional<Task> bestTask(const fs::path& fileName) {
  optional<Task> best;
  for (const string& line : File::openRead(fileName)) {
    optional<Task> task = parse(line);
    if (task && (!best || task->exponent < best->exponent)) { best = task; }
  }
  return best;
}

static string workName(i32 instance) { return "work-" + to_string(instance) + ".txt"; }

optional<Task> getWork(Args& args, i32 instance) {
  static mutex mut;

 again:
  // Try to get a task from the local work-<N> file.
  // This only reads from the per-worker file, so we don't need to lock.
  if (optional<Task> task = bestTask(workName(instance))) { return task; }

  // Try in order: the local worktodo.txt, and the global worktodo.txt if set up.
  vector<fs::path> worktodoFiles{"worktodo.txt"};
  if (!args.masterDir.empty()) { worktodoFiles.push_back(args.masterDir / "worktodo.txt"); }

  lock_guard lock(mut);

  for (fs::path& worktodo : worktodoFiles) {
    if (optional<Task> task = bestTask(worktodo)) {
      File::append(workName(instance), task->line);
      deleteLine(worktodo, task->line);
      goto again;
    }
  }

  return std::nullopt;
}

}

std::optional<Task> Worktodo::getTask(Args &args, i32 instance) {
  if (instance == 0) {
    if (args.prpExp) {
      u32 exp = args.prpExp;
      args.prpExp = 0;
      return Task{Task::PRP, exp};
    } else if (args.llExp) {
      u32 exp = args.llExp;
      args.llExp = 0;
      return Task{Task::LL, exp};
    } else if (!args.verifyPath.empty()) {
      auto path = args.verifyPath;
      args.verifyPath.clear();
      return Task{.kind=Task::VERIFY, .verifyPath=path};
    }
  }
  return getWork(args, instance);
}

bool Worktodo::deleteTask(const Task &task, i32 instance) {
  // Some tasks don't originate in worktodo.txt and thus don't need deleting.
  if (task.line.empty()) { return true; }
  return deleteLine(workName(instance), task.line);
}

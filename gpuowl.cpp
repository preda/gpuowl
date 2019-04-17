// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "Args.h"
#include "Task.h"
#include "Background.h"
#include "Worktodo.h"
#include "common.h"
#include "file.h"

#include <cstdio>
#include <filesystem>

extern string globalCpuName;

using namespace std::filesystem;

int main(int argc, char **argv) {
  initLog();
  log("%s %s\n", PROGRAM, VERSION);
  
  Background background;

  int exitCode = 0;
  
  try {
    Args args;
    string mainLine = Args::mergeArgs(argc, argv);
    args.parse(mainLine);
    if (!args.dir.empty()) { current_path(args.dir); }
    initLog("gpuowl.log");
    // log("Working directory: %s\n", string(current_path()).c_str());
    
    if (auto file = openRead("config.txt")) {
      char buf[256];
      while (fgets(buf, sizeof(buf), file.get())) {
        string line = buf;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) { line.pop_back(); }
        log("config: %s\n", line.c_str());
        args.parse(line);
      }
    } else {
      log("Note: no config.txt file found\n");
    }
    if (!args.cpu.empty()) { globalCpuName = args.cpu; }
    if (!mainLine.empty()) {
      log("config: %s\n", mainLine.c_str());
      args.parse(mainLine);
    }

    
    if (args.prpExp) {
      Worktodo::makePRP(args, args.prpExp).execute(args, background);      
    } else if (args.pm1Exp) {
      Worktodo::makePM1(args, args.pm1Exp).execute(args, background);
    } else {
      while (Task task = Worktodo::getTask(args)) {
        if (!task.execute(args, background)) { break; }
        Worktodo::deleteTask(task);
      }
    }
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
  }

  log("Bye\n");
  return exitCode; // not used yet.
}

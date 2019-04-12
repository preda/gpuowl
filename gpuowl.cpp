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
    args.parse(argc, argv);
    if (!args.dir.empty()) {
      path p = absolute(args.dir);
      current_path(p);
    }
    initLog("gpuowl.log");
    log("Working directory: %s\n", string(current_path()).c_str());
    
    if (auto file = openRead("config.txt")) {
      char line[256];
      while (fgets(line, sizeof(line), file.get())) { args.parse(line); }
    } else {
      log("Note: no config.txt file found\n");
    }
    if (!args.cpu.empty()) { globalCpuName = args.cpu; }
    args.parse(argc, argv);
    
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

// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "Args.h"
#include "Task.h"
#include "Background.h"
#include "worktodo.h"
#include "common.h"

extern string globalCpuName;

int main(int argc, char **argv) {  
  initLog("gpuowl.log");
  log("%s %s\n", PROGRAM, VERSION);
  
  Args args;
  if (!args.parse(argc, argv)) { return -1; }
  if (!args.cpu.empty()) { globalCpuName = args.cpu; }

  {
    string cmdLine;
    for (int i = 1; i < argc; ++i) { cmdLine += string(argv[i]) + " "; }
    log("%s\n", cmdLine.c_str());
  }

  Background background;
  
  try {
    while (Task task = Worktodo::getTask(args)) {
      if (!task.execute(args, background)) { break; }
      Worktodo::deleteTask(task);
    }
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
  }

  background.wait();
  
  log("Bye\n");
}

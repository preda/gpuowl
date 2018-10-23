// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "args.h"
#include "Task.h"
#include "Result.h"
#include "worktodo.h"
#include "common.h"

extern string globalCpuName;

int main(int argc, char **argv) {  
  initLog("gpuowl.log");
  log("%s %s\n", PROGRAM, VERSION);
  
  Args args;
  if (!args.parse(argc, argv)) { return -1; }
  if (!args.cpu.empty()) { globalCpuName = args.cpu; }

  try {
    while (Task task = Worktodo::getTask()) {
      task = task.morph(&args);
      if (!task.execute(args)) { break; }
      Worktodo::deleteTask(task);
    }
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
  }
  
  log("Bye\n");
}

// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "Args.h"
#include "Task.h"
#include "Background.h"
#include "Worktodo.h"
#include "common.h"

extern string globalCpuName;

int main(int argc, char **argv) {  
  initLog("gpuowl.log");
  log("%s %s\n", PROGRAM, VERSION);
  
  Args args;
  if (!args.parse(argc, argv)) { return -1; }
  if (!args.cpu.empty()) { globalCpuName = args.cpu; }

  Background background;

  int exitCode = 0;
  
  try {
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

  background.wait();
  
  log("Bye\n");
  return exitCode; // not used yet.
}

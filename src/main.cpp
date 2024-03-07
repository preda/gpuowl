// GpuOwl Mersenne primality tester
// Copyright (C) Mihai Preda

#include "Args.h"
#include "Task.h"
#include "Worktodo.h"
#include "version.h"
#include "AllocTrac.h"
#include "typeName.h"
#include "log.h"

#include <filesystem>

extern string globalCpuName;

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  initLog();
  log("PRPLL %s\n", VERSION);
  
  int exitCode = 0;

  try {
    string mainLine = Args::mergeArgs(argc, argv);
    {
      Args args{true};
      args.parse(mainLine);
      if (!args.dir.empty()) {
        fs::current_path(args.dir);
      }
    }
    
    fs::path poolDir;
    {
      Args args{true};
      args.readConfig("config.txt");
      args.parse(mainLine);
      poolDir = args.masterDir;
      globalCpuName = args.cpu;
    }
    
    Args args;
    
    initLog((poolDir / "gpuowl.log").string().c_str());
    log("PRPLL %s\n", VERSION);
    
    if (!poolDir.empty()) { args.readConfig(poolDir / "config.txt"); }
    args.readConfig("config.txt");
    args.parse(mainLine);
    args.setDefaults();
        
    if (args.maxAlloc) { AllocTrac::setMaxAlloc(args.maxAlloc); }
    
    if (args.prpExp) {
      Worktodo::makePRP(args.prpExp).execute(args);
    } else if (args.llExp) {
      Worktodo::makeLL(args.llExp).execute(args);
    } else if (!args.verifyPath.empty()) {
      Worktodo::makeVerify(args, args.verifyPath).execute(args);
    } else {
      while (auto task = Worktodo::getTask(args)) { task->execute(args); }
    }
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
  } catch (const string& mes) {
    log("Exiting because \"%s\"\n", mes.c_str());
  }
  /*catch (const std::exception& e) {
    log("Exception %s: %s\n", typeName(e), e.what());
    throw;
  } catch (...) {
    log("Unexpected exception\n");
    throw;
  }
  */

  // background.wait();
  // if (factorFoundForExp) { Worktodo::deletePRP(factorFoundForExp); }
  log("Bye\n");
  return exitCode; // not used yet.
}

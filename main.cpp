// GpuOwl Mersenne primality tester; Copyright Mihai Preda.

#include "Args.h"
#include "Task.h"
#include "Background.h"
#include "Worktodo.h"
#include "common.h"
#include "File.h"
#include "version.h"
#include "AllocTrac.h"
#include "typeName.h"

#include <cstdio>
#include <filesystem>

extern string globalCpuName;

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  initLog();
  log("gpuowl %s\n", VERSION);
  
  Background background;

  int exitCode = 0;
  
  try {
    string mainLine = Args::mergeArgs(argc, argv);
    {
      Args args;
      args.parse(mainLine);
      if (!args.dir.empty()) { fs::current_path(args.dir); }
      initLog("gpuowl.log");
    }

    Args args;
    if (auto file = File::openRead("config.txt")) {
      while (true) {
        if (string line = file.readLine(); !line.empty()) {
          line = rstripNewline(line);
          log("config.txt: %s\n", line.c_str());
          args.parse(line);        
        } else {
          break;
        }
      }
    } else {
      log("Note: no config.txt file found\n");
    }
    
    if (!mainLine.empty()) {
      log("config: %s\n", mainLine.c_str());
    }
    args.parse(mainLine);
    args.setDefaults();
    if (!args.cpu.empty()) { globalCpuName = args.cpu; }
    
    if (args.maxAlloc) { AllocTrac::setMaxAlloc(args.maxAlloc); }
    
    if (args.prpExp) {
      Worktodo::makePRP(args, args.prpExp).execute(args, background);      
    } else if (args.pm1Exp) {
      Worktodo::makePM1(args, args.pm1Exp).execute(args, background);
    } else {
      while (auto task = Worktodo::getTask(args)) {
        if (!task->execute(args, background)) { break; }
        Worktodo::deleteTask(*task);
      }
    }
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
  } catch (const std::exception& e) {
    log("Exception %s: %s\n", typeName(e), e.what());
  } catch (...) {
    log("Unexpected exception\n");
  }

  background.wait();
  log("Bye\n");
  return exitCode; // not used yet.
}

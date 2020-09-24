// GpuOwl Mersenne primality tester; Copyright Mihai Preda.

#include "Args.h"
#include "Task.h"
// #include "Background.h"
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

static void readConfig(Args& args, const fs::path& path, bool doLog) {
  if (auto file = File::openRead(path)) {
    // log("reading %s\n", path.c_str());
    while (true) {
      if (string line = file.readLine(); !line.empty()) {
        line = rstripNewline(line);
        if (doLog) { log("config: %s\n", line.c_str()); }
        args.parse(line);
      } else {
        break;
      }
    }
  } else {
    if (doLog) { log("Note: not found '%s'\n", path.string().c_str()); }
  }
}

int main(int argc, char **argv) {
  initLog();
  log("gpuowl %s\n", VERSION);
  // Background background;

  int exitCode = 0;

  try {
    string mainLine = Args::mergeArgs(argc, argv);
    {
      Args args;
      args.parse(mainLine);
      if (!args.dir.empty()) { fs::current_path(args.dir); }
      initLog("gpuowl.log");
    }

    fs::path poolDir = [&mainLine](){
                         Args args;
                         readConfig(args, "config.txt", false);
                         args.parse(mainLine);
                         return args.masterDir;
                       }();

    Args args;
    if (!poolDir.empty()) { readConfig(args, poolDir / "config.txt", true); }
    readConfig(args, "config.txt", true);
    if (!mainLine.empty()) {
      log("config: %s\n", mainLine.c_str());
      args.parse(mainLine);
    }
    args.setDefaults();
    if (!args.cpu.empty()) { globalCpuName = args.cpu; }
    
    if (args.maxAlloc) { AllocTrac::setMaxAlloc(args.maxAlloc); }
    
    if (args.prpExp) {
      Worktodo::makePRP(args, args.prpExp).execute(args);
    } else if (!args.verifyPath.empty()) {
      Worktodo::makeVerify(args, args.verifyPath).execute(args);
    } else {
      while (auto task = Worktodo::getTask(args)) { task->execute(args); }
    }
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
  } catch (const std::exception& e) {
    log("Exception %s: %s\n", typeName(e), e.what());
  } catch (...) {
    log("Unexpected exception\n");
  }

  // background.wait();
  // if (factorFoundForExp) { Worktodo::deletePRP(factorFoundForExp); }
  log("Bye\n");
  return exitCode; // not used yet.
}

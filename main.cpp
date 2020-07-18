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

void readConfig(Args& args, const std::string& path, bool doLog) {
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
    if (doLog) { log("Note: not found '%s'\n", path.c_str()); }
  }
}

int main(int argc, char **argv) {
  initLog();
  log("gpuowl %s\n", VERSION);
  // log("%s %s\n", MD5::hash(""s).c_str(), MD5::hash("The quick brown fox jumps over the lazy dog"s).c_str());
  
  
  Background background;

  int exitCode = 0;

  std::atomic<u32> factorFoundForExp = 0;
  
  try {
    string mainLine = Args::mergeArgs(argc, argv);
    {
      Args args;
      args.parse(mainLine);
      if (!args.dir.empty()) { fs::current_path(args.dir); }
      initLog("gpuowl.log");
    }

    string poolDir{};
    {
      Args args;
      readConfig(args, "config.txt", false);
      args.parse(mainLine);
      poolDir = args.masterDir;
    }

    Args args;
    if (!poolDir.empty()) { readConfig(args, poolDir + '/' + "config.txt", true); }
    readConfig(args, "config.txt", true);
    if (!mainLine.empty()) {
      log("config: %s\n", mainLine.c_str());
      args.parse(mainLine);
    }
    args.setDefaults();
    if (!args.cpu.empty()) { globalCpuName = args.cpu; }
    
    if (args.maxAlloc) { AllocTrac::setMaxAlloc(args.maxAlloc); }
    
    if (args.prpExp) {
      Worktodo::makePRP(args, args.prpExp).execute(args, background, factorFoundForExp);
    } else if (args.pm1Exp) {
      Worktodo::makePM1(args, args.pm1Exp).execute(args, background, factorFoundForExp);
    } else if (args.llExp) {
      Worktodo::makeLL(args, args.llExp).execute(args, background, factorFoundForExp);
    } else if (!args.verifyPath.empty()) {
      Worktodo::makeVerify(args, args.verifyPath).execute(args, background, factorFoundForExp);
    } else {
      while (auto task = Worktodo::getTask(args)) { task->execute(args, background, factorFoundForExp); }
    }
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
  } catch (const std::exception& e) {
    log("Exception %s: %s\n", typeName(e), e.what());
  } catch (...) {
    log("Unexpected exception\n");
  }

  background.wait();
  if (factorFoundForExp) { Worktodo::deletePRP(factorFoundForExp); }
  log("Bye\n");
  return exitCode; // not used yet.
}

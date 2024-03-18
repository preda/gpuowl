// GpuOwl Mersenne primality tester
// Copyright (C) Mihai Preda

#include "Args.h"
#include "Queue.h"
#include "Task.h"
#include "Worktodo.h"
#include "version.h"
#include "AllocTrac.h"
#include "typeName.h"
#include "log.h"
#include "Context.h"

#include <filesystem>
#include <thread>

extern string globalCpuName;

namespace fs = std::filesystem;

void gpuWorker(Args& args, Queue *q, i32 instance) {
  log("Starting worker %d at thread %d\n", instance, Queue::registerThread());
  try {
    while (auto task = Worktodo::getTask(args, instance)) { task->execute(q, args); }
  } catch (const char *mes) {
    log("Exception \"%s\"\n", mes);
  } catch (const string& mes) {
    log("Exception \"%s\"\n", mes.c_str());
  } catch (const std::exception& e) {
    log("Exception %s: %s\n", typeName(e), e.what());
  }
}

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
    
    Context context(getDevice(args.device));
    Queue queue{args, context};
    jthread t2{gpuWorker, ref(args), &queue, 1};
    gpuWorker(args, &queue, 0);
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
  } catch (const string& mes) {
    log("Exiting because \"%s\"\n", mes.c_str());
  }

  log("Bye\n");
  return exitCode; // not used yet.
}

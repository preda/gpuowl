// GpuOwl Mersenne primality tester
// Copyright (C) Mihai Preda

#include "Args.h"
#include "Queue.h"
#include "Signal.h"
#include "Task.h"
#include "Worktodo.h"
#include "version.h"
#include "AllocTrac.h"
#include "typeName.h"
#include "log.h"
#include "Context.h"
#include "TrigBufCache.h"

#include <filesystem>
#include <thread>
#include <cstdlib>

namespace fs = std::filesystem;

void gpuWorker(Args& args, Queue *q, TrigBufCache* bufCache, i32 instance) {
  LogContext context{(instance ? args.cpu : ""s) + to_string(instance) + ' '};
  // log("Starting worker %d\n", instance);
  try {
    while (auto task = Worktodo::getTask(args, instance)) { task->execute(q, args, bufCache); }
  } catch (const char *mes) {
    log("Exception \"%s\"\n", mes);
  } catch (const string& mes) {
    log("Exception \"%s\"\n", mes.c_str());
  } catch (const std::exception& e) {
    log("Exception %s: %s\n", typeName(e), e.what());
  }
}

unique_ptr<LogContext> cpuNameContext;

int main(int argc, char **argv) {
  // Required to work around a ROCm bug when using multiple queues
  setenv("ROC_SIGNAL_POOL_SIZE", "32", 0);

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
      cpuNameContext = make_unique<LogContext>(args.cpu);
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
    TrigBufCache bufCache{&context};
    Signal signal;

    vector<Queue> queues;
    for (int i = 0; i < int(args.workers); ++i) { queues.emplace_back(args, context); }
    vector<jthread> threads;
    for (int i = 1; i < int(args.workers); ++i) {
      threads.emplace_back(gpuWorker, ref(args), &queues[i], &bufCache, i);
    }
    gpuWorker(args, &queues[0], &bufCache, 0);
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
  } catch (const string& mes) {
    log("Exiting because \"%s\"\n", mes.c_str());
  }

  log("Bye\n");
  cpuNameContext.reset();
  return exitCode; // not used yet.
}

// Copyright (C) Mihai Preda

#include "Args.h"
#include "Background.h"
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
#include "GpuCommon.h"
#include "Gpu.h"
#include "tune.h"

#include <filesystem>
#include <thread>

namespace fs = std::filesystem;

void gpuWorker(GpuCommon shared, Queue *q, i32 instance) {
  LogContext context{(instance ? shared.args->tailDir() : ""s) + to_string(instance) + ' '};
  // log("Starting worker %d\n", instance);
  try {
    while (auto task = Worktodo::getTask(*shared.args, instance)) { task->execute(shared, q, instance); }
  } catch (const char *mes) {
    log("Exception \"%s\"\n", mes);
  } catch (const string& mes) {
    log("Exception \"%s\"\n", mes.c_str());
  } catch (const std::exception& e) {
    log("Exception %s: %s\n", typeName(e), e.what());
  }
}


#ifdef __MINGW32__ // for Windows
extern int putenv(const char *);
#endif

int main(int argc, char **argv) {

#ifdef __MINGW32__
  putenv("ROC_SIGNAL_POOL_SIZE=32");
#else
  // Required to work around a ROCm bug when using multiple queues
  setenv("ROC_SIGNAL_POOL_SIZE", "32", 0);
#endif

  unique_ptr<LogContext> cpuNameContext;

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
      cpuNameContext = make_unique<LogContext>(args.tailDir());
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
    Background background;
    GpuCommon shared{&args, &bufCache, &background};

    if (!args.ctune.empty() || args.doTune || args.doZtune || args.carryTune) {
      Queue q(context, args.profile);
      Tune tune{&q, shared};

      if (!args.ctune.empty()) {
        tune.ctune();
      } else if (args.doTune) {
        tune.tune();
      } else if (args.doZtune) {
        tune.ztune();
      } else if (args.carryTune) {
        tune.carryTune();
      }
    } else {
      {
        vector<Queue> queues;
        for (int i = 0; i < int(args.workers); ++i) { queues.emplace_back(context, args.profile); }
        vector<jthread> threads;
        for (int i = 1; i < int(args.workers); ++i) {
          threads.emplace_back(gpuWorker, shared, &queues[i], i);
        }
        gpuWorker(shared, &queues[0], 0);
      }

      log("No more work. Add work to worktodo.txt , see -h for details.\n");
    }
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
  } catch (const string& mes) {
    log("Exiting because \"%s\"\n", mes.c_str());
  }

  log("Bye\n");
  return exitCode; // not used yet.
}

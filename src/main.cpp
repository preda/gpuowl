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

#include <filesystem>
#include <thread>
#include <cinttypes>

namespace {

vector<string> split(const string& s, char delim) {
  vector<string> ret;
  size_t start = 0;
  while (true) {
    size_t p = s.find(delim, start);
    if (p == string::npos) {
      ret.push_back(s.substr(start));
      break;
    } else {
      ret.push_back(s.substr(start, p - start));
    }
    start = p + 1;
  }
  return ret;
}

/*
vector<u32> splitInts(const string& s, char delim) {
  vector<u32> ret;
  for (string& x : split(s, delim)) {
    ret.push_back(stoul(x));
  }
  return ret;
}
*/

using TuneConfig = vector<pair<string, string>>;

vector<TuneConfig> getTuneConfigs(const string& tune) {
  vector<pair<string, vector<string>>> params;
  for (auto& part : split(tune, ';')) {
    // log("part '%s'\n", part.c_str());
    auto keyVal = split(part, '=');
    assert(keyVal.size() == 2);
    string key = keyVal.front();
    string val = keyVal.back();
    // log("k '%s' v '%s'\n", key.c_str(), val.c_str());
    params.push_back({key, split(val, ',')});
  }

  vector<vector<pair<string, string>>> configs;

  int n = params.size();
  vector<int> vpos(n);
  while (true) {
    vector<pair<string, string>> config;
    for (int i = 0; i < n; ++i) {
      config.push_back({params[i].first, params[i].second[vpos[i]]});
    }
    configs.push_back(config);

    int i;
    for (i = n-1; i >= 0; --i) {
      if (vpos[i] < int(params[i].second.size()) - 1) {
        ++vpos[i];
        break;
      } else {
        vpos[i] = 0;
      }
    }

    if (i < 0) { return configs; }
  }
}

string toString(const vector<pair<string, string>>& config) {
  string s;
  for (auto [k, v] : config) { s += k + '=' + v + ' '; }
  return s;
}

}

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

    if (!args.tune.empty()) {
      Queue q(context, args.profile);

      auto configs = getTuneConfigs(args.tune);
      vector<pair<double, string>> results;

      for (const auto& config : configs) {
        u32 exponent = 0;
        // log("Timing %s\n", toString(config).c_str());
        for (auto& [k, v] : config) {
          if (k == "fft") {
            args.fftSpec = v;
          } if (k == "E") {
            exponent = stoll(v);
          } else {
            assert(k == "IN_WG" || k == "OUT_WG" || k == "IN_SIZEX" || k == "OUT_SIZEX" || k == "OUT_SPACING");
            args.flags[k] = v;
          }
        }
        if (!exponent) {
          log("No exponent in tune\n");
          throw "The exponent E=<N> must be set in tune=<values>";
        }
        auto gpu = Gpu::make(&q, exponent, shared, false);
        auto [secsPerIt, res64] = gpu->timePRP();
        if (secsPerIt < 0) {
          log("Error %016" PRIx64 " %s\n", res64, toString(config).c_str());
        } else {
          log("%.1f us/it  %016" PRIx64 " %s\n", secsPerIt * 1e6, res64, toString(config).c_str());
          results.push_back({secsPerIt, toString(config)});
        }
      }

      log("Tune top results:\n");
      std::sort(results.begin(), results.end());
      for (int i = 0; i < 20 && i < int(results.size()); ++i) {
        log("%2d %.1f %s\n", i, results[i].first, results[i].second.c_str());
      }
    } else {
      vector<Queue> queues;
      for (int i = 0; i < int(args.workers); ++i) { queues.emplace_back(context, args.profile); }

      vector<jthread> threads;
      for (int i = 1; i < int(args.workers); ++i) {
        threads.emplace_back(gpuWorker, shared, &queues[i], i);
      }
      gpuWorker(shared, &queues[0], 0);
    }
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
  } catch (const string& mes) {
    log("Exiting because \"%s\"\n", mes.c_str());
  }

  log("Bye\n");
  return exitCode; // not used yet.
}

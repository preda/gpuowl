// Copyright Mihai Preda.

#pragma once

#include "clpp.h"
#include "timeutil.h"
#include "common.h"

#include <string>

struct TimeInfo {
  double total = 0;
  u32 n = 0;

  void add(double deltaTime, u32 deltaN = 1) { total += deltaTime; n += deltaN; }
  void reset() { total = 0; n = 0; }
};

class Kernel {
  KernelHolder kernel;
  Queue queue;
  int nWorkGroups;
  string name;
  bool doTime;
  int groupSize;
  TimeInfo stats;
  
public:
  Kernel(cl_program program, Queue queue, cl_device_id device, int nWorkGroups, const std::string &name, bool doTime) :
    kernel(makeKernel(program, name.c_str())),
    queue(queue),
    nWorkGroups(nWorkGroups),
    name(name),
    doTime(doTime),
    groupSize(getWorkGroupSize(kernel.get(), device, name.c_str()))
  {}

  template<typename... Args> void setFixedArgs(int pos, const Args &...tail) { setArgs(pos, tail...); }
  
  template<typename... Args> void operator()(const Args &...args) {
    setArgs(0, args...);
    run();
  }

  string getName() { return name; }

  TimeInfo resetStats() { auto ret = stats; stats.reset(); return ret; }

private:
  template<typename T> void setArgs(int pos, const T &arg) { ::setArg(kernel.get(), pos, arg); }
  
  template<typename T, typename... Args> void setArgs(int pos, const T &arg, const Args &...tail) {
    setArgs(pos, arg);
    setArgs(pos + 1, tail...);
  }
  
  void run() {
    if (doTime) {
      // queue.finish();
      Timer timer;
      queue.run(kernel.get(), groupSize, nWorkGroups * groupSize, name);
      queue.finish();
      stats.add(timer.deltaMicros());
    } else {
      queue.run(kernel.get(), groupSize, nWorkGroups * groupSize, name);
    }
  }
};

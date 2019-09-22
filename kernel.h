// Copyright Mihai Preda.

#pragma once

#include "Queue.h"
#include "timeutil.h"
#include "common.h"

#include <string>

class Kernel {
  KernelHolder kernel;
  QueuePtr queue;
  int nWorkGroups;
  string name;
  int groupSize;
  
public:
  Kernel(cl_program program, QueuePtr queue, cl_device_id device, int nWorkGroups, const std::string &name) :
    kernel(makeKernel(program, name.c_str())),
    queue(std::move(queue)),
    nWorkGroups(nWorkGroups),
    name(name),
    groupSize(getWorkGroupSize(kernel.get(), device, name.c_str()))
  {}

  template<typename... Args> void setFixedArgs(int pos, const Args &...tail) { setArgs(pos, tail...); }
  
  template<typename... Args> void operator()(const Args &...args) {
    setArgs(0, args...);
    run();
  }

  string getName() { return name; }

private:
  template<typename T> void setArgs(int pos, const T &arg) { ::setArg(kernel.get(), pos, arg); }
  
  template<typename T, typename... Args> void setArgs(int pos, const T &arg, const Args &...tail) {
    setArgs(pos, arg);
    setArgs(pos + 1, tail...);
  }
  
  void run() { queue->run(kernel.get(), groupSize, nWorkGroups * groupSize, name); }
};

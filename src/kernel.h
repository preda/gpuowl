// Copyright (C) Mihai Preda.

#pragma once

#include "Queue.h"
#include "Buffer.h"
#include "timeutil.h"
#include "common.h"

#include <string>
#include <stdexcept>

class KernelCompiler;

class Kernel {
  const string name;
  const string fileName;
  const string nameInFile;
  const string defines;
  
  QueuePtr queue;
  size_t workSize;
  u32 groupSize{};
  
  KernelHolder kernel{};

public:
  Kernel(string_view name, QueuePtr queue,
         string_view fileName, string_view nameInFile,
         string_view defines,
         u32 groupSize, size_t workSize):
    name{name},
    fileName{fileName},
    nameInFile{nameInFile},
    defines{defines},
    queue{queue},
    workSize{workSize},
    groupSize{groupSize}
  {
    assert(workSize % groupSize == 0);
  }

  void load(const KernelCompiler& compiler, cl_device_id deviceId);
  
  void init(KernelHolder kern) {
    assert(kern);
    kernel = std::move(kern);
    // groupSize = getWorkGroupSize(kernel.get(), device, name.c_str());
  }
    
  /*
  Kernel(cl_program program, QueuePtr queue, cl_device_id device, u32 nWorkGroups, const std::string &name) :
    kernel(makeKernel(program, name.c_str())),
    groupSize(kernel ? getWorkGroupSize(kernel.get(), device, name.c_str()) : 0),
    queue(std::move(queue)),
    workSize(nWorkGroups * groupSize),
    name(name)
  {}

  Kernel(cl_program program, QueuePtr queue, cl_device_id device, const std::string &name, size_t workSize) :
    kernel(makeKernel(program, name.c_str())),
    groupSize(kernel ? getWorkGroupSize(kernel.get(), device, name.c_str()) : 0),
    queue(std::move(queue)),
    workSize(workSize),
    name(name)
  {
    assert(groupSize == 0 || (workSize % groupSize == 0));
  }
  */

  
  template<typename... Args> void setFixedArgs(int pos, const Args &...tail) { setArgs(name, pos, tail...); }
  
  template<typename... Args> void operator()(const Args &...args) {
    setArgs(name, 0, args...);
    run();
  }

  string getName() { return name; }

private:
  template<typename T> void setArgs(const string& name, int pos, const ConstBuffer<T>& buf) { setArgs(name, pos, buf.get()); }
  template<typename T> void setArgs(const string& name, int pos, const Buffer<T>& buf) { setArgs(name, pos, buf.get()); }
  template<typename T> void setArgs(const string& name, int pos, const HostAccessBuffer<T>& buf) { setArgs(name, pos, buf.get()); }
  template<typename T> void setArgs(const string& name, int pos, const T &arg) { ::setArg(kernel.get(), pos, arg, name); }
  
  template<typename T, typename... Args> void setArgs(const string& name, int pos, const T &arg, const Args &...tail) {
    setArgs(name, pos, arg);
    setArgs(name, pos + 1, tail...);
  }
  
  void run() {
    if (kernel) {
      queue->run(kernel.get(), groupSize, workSize, name);
    } else {
      throw std::runtime_error("OpenCL kernel "s + name + " not found");
    }
  }
};

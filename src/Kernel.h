// Copyright (C) Mihai Preda.

#pragma once

#include "Queue.h"
#include "Buffer.h"
#include "common.h"

#include <future>
#include <string>
#include <vector>
#include <utility>

class KernelCompiler;
class TimeInfo;

class Kernel {
  const string name;
  const string fileName;
  const string nameInFile;
  const string defines;
  
  TimeInfo *timeInfo;

  Queue* queue;
  size_t workSize;
  u32 groupSize = 0;
  
  KernelHolder kernel{};
  std::future<KernelHolder> pendingKernel;
  cl_device_id deviceId;
  std::vector<std::pair<u32, cl_mem>> pendingArgs;

public:
  Kernel(string_view name, TimeInfo* timeInfo, Queue* queue,
         string_view fileName, string_view nameInFile,
         size_t workSize, string_view defines = "");

  ~Kernel();

  void startLoad(const KernelCompiler& compiler);
  void finishLoad();
  
  template<typename... Args> void setFixedArgs(int pos, const Args &...tail) { setArgs(pos, tail...); }
  
  template<typename... Args> void operator()(const Args &...args) {
    setArgs(0, args...);
    run();
  }

private:
  template<typename T> void setArgs(int pos, const shared_ptr<Buffer<T>>& buf) { setArgs(pos, buf->get()); }
  template<typename T> void setArgs(int pos, const Buffer<T>* buf) { setArgs(pos, buf->get()); }
  template<typename T> void setArgs(int pos, const Buffer<T>& buf) { setArgs(pos, buf.get()); }

  void setArgs(int pos, cl_mem arg) {
    if (kernel) {
      ::setArg(kernel.get(), pos, arg, name);
    } else {
      pendingArgs.push_back({pos, arg});
    }
  }

  template<typename T> void setArgs(int pos, const T &arg) { ::setArg(kernel.get(), pos, arg, name); }
  
  template<typename T, typename... Args> void setArgs(int pos, const T &arg, const Args &...tail) {
    setArgs(pos, arg);
    setArgs(pos + 1, tail...);
  }
  
  void run();
};

// Copyright (C) Mihai Preda.

#pragma once

#include "Queue.h"
#include "Buffer.h"
#include "common.h"

#include <string>

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

public:
  Kernel(string_view name, TimeInfo* timeInfo, Queue* queue,
         string_view fileName, string_view nameInFile,
         size_t workSize, string_view defines = "");

  ~Kernel();

  void load(const KernelCompiler& compiler);
  
  template<typename... Args> void setFixedArgs(int pos, const Args &...tail) { setArgs(name, pos, tail...); }
  
  template<typename... Args> void operator()(const Args &...args) {
    setArgs(name, 0, args...);
    run();
  }

private:
  template<typename T> void setArgs(const string& name, int pos, const Buffer<T>& buf) { setArgs(name, pos, buf.get()); }
  template<typename T> void setArgs(const string& name, int pos, const T &arg) { ::setArg(kernel.get(), pos, arg, name); }
  
  template<typename T, typename... Args> void setArgs(const string& name, int pos, const T &arg, const Args &...tail) {
    setArgs(name, pos, arg);
    setArgs(name, pos + 1, tail...);
  }
  
  void run();
};

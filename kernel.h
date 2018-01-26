#pragma once

#include "clwrap.h"
#include "common.h"

#include <string>
#include <vector>
#include <memory>

template<typename T>
struct ReleaseDelete {
  using pointer = T;
  
  void operator()(T t) {
    // fprintf(stderr, "Release %s %llx\n", typeid(T).name(), u64(t));
    release(t);
  }
};

template<typename T> using Holder = std::unique_ptr<T, ReleaseDelete<T> >;

using Buffer  = Holder<cl_mem>;
using Context = Holder<cl_context>;
using Queue   = Holder<cl_queue>;

static_assert(sizeof(Buffer) == sizeof(cl_mem), "size Buffer");

class Kernel {
  Holder<cl_kernel> kernel;
  cl_queue queue;
  int N;
  int itemsPerThread;
  int nArgs;
  std::string name;
  std::vector<std::string> argNames;
  u64 timeSum;
  bool doTime;

  int getArgPos(const std::string &name) {
    for (int i = 0; i < nArgs; ++i) { if (argNames[i] == name) { return i; } }
    return -1;
  }
  
public:
  Kernel(cl_program program, cl_queue q, int N, const std::string &name, int itemsPerThread, bool doTime) :
    kernel(makeKernel(program, name.c_str())),
    queue(q),
    N(N),
    itemsPerThread(itemsPerThread),
    nArgs(getKernelNumArgs(kernel.get())),
    name(name),
    timeSum(0),
    doTime(doTime)
  {
    assert(N % itemsPerThread == 0);
    assert(nArgs >= 0);
    for (int i = 0; i < nArgs; ++i) { argNames.push_back(getKernelArgName(kernel.get(), i)); }
    // log("kernel %s: %d args, arg0 %s\n", name.c_str(), getKernelNumArgs(kernel.get()), getKernelArgName(kernel.get(), 0).c_str());
    // log("kernel %s, queue %p\n", name, q);
  }
  
  void operator()() {
    if (doTime) {
      Timer timer;
      ::run(queue, kernel.get(), N / itemsPerThread, name);
      finish();
      timeSum += timer.deltaMicros();
    } else {
      ::run(queue, kernel.get(), N / itemsPerThread, name);
    }
  }
  
  string getName() { return name; }
  
  void setArg(const std::string &name, const auto &arg) {
    int pos = getArgPos(name);
    assert(pos >= 0);
    ::setArg(kernel.get(), pos, arg);
  }
  
  void setArg(const std::string &name, const Buffer &buf) { setArg(name, buf.get()); }
  
  void finish() { ::finish(queue); }

  u64 getTime() { return timeSum; }
  void resetTime() { timeSum = 0; }
};

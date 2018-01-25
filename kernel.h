#pragma once

#include "clwrap.h"
#include "common.h"

#include <string>

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
  template<int P> void setArgsAt() {}  
  template<int P> void setArgsAt(auto &a, auto&... args) {
    setArg(P, a);
    setArgsAt<P + 1>(args...);
  }

public:
  virtual ~Kernel() {};
  virtual void run(cl_queue q) = 0;
  virtual string getName() = 0;
  virtual cl_kernel getKernel() = 0;
  
  void setArg(int pos, const auto &arg) { ::setArg(getKernel(), pos, arg); }
  void setArg(int pos, const Buffer &buf) { setArg(pos, buf.get()); }
  void setArgs(const auto&... args) { setArgsAt<0>(args...); }
};

class BaseKernel : public Kernel {
  Holder<cl_kernel> kernel;
  int N;
  int itemsPerThread;
  std::string name;

public:
  BaseKernel(cl_program program, int N, const std::string &name, int itemsPerThread) :
    kernel(makeKernel(program, name.c_str())),
    N(N),
    itemsPerThread(itemsPerThread),
    name(name)
  {
    assert(N % itemsPerThread == 0);
    // log("kernel %s: %d args, arg0 %s\n", name.c_str(), getKernelNumArgs(kernel.get()), getKernelArgName(kernel.get(), 0).c_str());
  }

  virtual void run(cl_queue q) { ::run(q, kernel.get(), N / itemsPerThread, name); }

  virtual string getName() { return name; }

  virtual cl_kernel getKernel() { return kernel.get(); }
};

class TimedKernel : public Kernel {
  std::unique_ptr<Kernel> kernel;
  u64 timeAcc;

public:
  TimedKernel(Kernel *k) : kernel(k), timeAcc(0) { }

  virtual void run(cl_queue q) {
    Timer timer;
    kernel->run(q);
    finish(q);
    timeAcc += timer.deltaMicros();
  }

  virtual string getName() { return kernel->getName(); }
  virtual cl_kernel getKernel() { return kernel->getKernel(); }
  
  u64 getTime() { return timeAcc; }
  void resetTime() { timeAcc = 0; }
};

/*
class CompoundKernel : public Kernel {
  std::vector<Kernel> kernels;

public:
  CompoundKernel(std::vector<Kernel> kernels) :
    kernels(kernels) {
  }

  void run(cl_queue q) { for (Kerner k : kernels) { k->run(q); } }

  string getName() { 
    
  
};
*/

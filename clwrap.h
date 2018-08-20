// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "tinycl.h"

#include <string>
#include <vector>
#include <cassert>

using std::string;
using std::vector;

typedef cl_command_queue cl_queue;

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
using QueueHolder = Holder<cl_queue>;

static_assert(sizeof(Buffer) == sizeof(cl_mem), "size Buffer");

const unsigned BUF_CONST = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
const unsigned BUF_RW    = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;

bool check(int err, const char *mes = nullptr);
#define CHECK(what) assert(check(what));

vector<cl_device_id> getDeviceIDs(bool onlyGPU);
string getShortInfo(cl_device_id device);
string getLongInfo(cl_device_id device);
cl_context createContext(cl_device_id device);
void release(cl_context context);
void release(cl_program program);
void release(cl_mem buf);
void release(cl_queue queue);
void release(cl_kernel k);
cl_program compile(cl_device_id device, cl_context context, const string &name, const string &extraArgs,
                   const vector<pair<string, unsigned>> &defines, const string &config = "");
cl_kernel makeKernel(cl_program program, const char *name);

template<typename T>
void setArg(cl_kernel k, int pos, const T &value) { CHECK(clSetKernelArg(k, pos, sizeof(value), &value)); }
// Special-case Buffer argument: pass the wrapped cl_mem.
void setArg(cl_kernel k, int pos, const Buffer &buf);
cl_mem makeBuf(cl_context context, unsigned kind, size_t size, const void *ptr = 0);
cl_mem makeBuf(Context &context, unsigned kind, size_t size, const void *ptr = 0);
cl_queue makeQueue(cl_device_id d, cl_context c);

void flush( cl_queue q);
void finish(cl_queue q);

void run(cl_queue queue, cl_kernel kernel, size_t groupSize, size_t workSize, const string &name);
void read(cl_queue queue, bool blocking, cl_mem buf, size_t size, void *data, size_t start = 0);
void read(cl_queue queue, bool blocking, Buffer &buf, size_t size, void *data, size_t start = 0);
void write(cl_queue queue, bool blocking, cl_mem buf, size_t size, const void *data, size_t start = 0);
void write(cl_queue queue, bool blocking, Buffer &buf, size_t size, const void *data, size_t start = 0);
void copyBuf(cl_queue queue, Buffer &src, Buffer &dst, size_t size);
int getKernelNumArgs(cl_kernel k);
int getWorkGroupSize(cl_kernel k, cl_device_id device, const char *name);
std::string getKernelArgName(cl_kernel k, int pos);

class Queue {
  QueueHolder queue;
  
public:
  Queue(cl_queue queue) : queue(queue) {}

  template<typename T> vector<T> read(Buffer &buf, size_t nItems) {
    vector<T> ret(nItems);
    ::read(queue.get(), true, buf, nItems * sizeof(T), ret.data());
    return ret;
  }

  template<typename T> void write(Buffer &buf, const vector<T> &vect) {
    ::write(queue.get(), true, buf, vect.size() * sizeof(T), vect.data());
  }

  template<typename T> void copy(Buffer &src, Buffer &dst, size_t nItems) {
    ::copyBuf(queue.get(), src, dst, nItems * sizeof(T));
  }
  
  void run(cl_kernel kernel, size_t groupSize, size_t workSize, const string &name) {
    ::run(queue.get(), kernel, groupSize, workSize, name);
  }

  void finish() { ::finish(queue.get()); }

  cl_queue get() { return queue.get(); }

  void zero(Buffer &buf, size_t size);
};

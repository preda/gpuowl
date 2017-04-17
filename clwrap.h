// Copyright (C) 2017 Mihai Preda.

#include "tinycl.h"
#include <cassert>
#include <cstdio>
#include <cstdarg>
#include <sys/stat.h>
#include <sys/time.h>

long timeMillis() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

#define CHECK(err) { int e = err; if (e != CL_SUCCESS) { fprintf(stderr, "error %d\n", e); assert(false); }}

#define CHECK2(err, mes) { int e = err; if (e != CL_SUCCESS) { fprintf(stderr, "error %d (%s)\n", e, mes); assert(false); }}

class Context {
public:
  cl_device_id device;
  cl_context context;
  
  Context() {
    cl_platform_id platform;
    CHECK(clGetPlatformIDs(1, &platform, NULL));
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
    int err;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CHECK(err);
  }

  ~Context() {
    CHECK(clReleaseContext(context));
  }
};

class Program {
public:
  cl_program program;

  Program(Context &c, const char *f, const char *opts = "") { compile(c, f, opts); }
  ~Program() { clReleaseProgram(program); }
  
  void compile(Context &c, const char *fileName, const char *extra) {
    FILE *f = fopen(fileName, "r");
    assert(f);
    
    char buf[256 * 1024];
    size_t size = read(buf, sizeof(buf), f);
    assert(size < sizeof(buf));
    char *pbuf = buf;
    int err;
    program = clCreateProgramWithSource(c.context, 1, (const char **)&pbuf, &size, &err);
    CHECK(err);
    mkdir("isa", 0777);
    snprintf(buf, sizeof(buf),
             "%s -Werror -cl-fast-relaxed-math -cl-std=CL2.0 -cl-uniform-work-group-size -I. -fno-bin-llvmir -save-temps=isa/", extra);

    err = clBuildProgram(program, 1, &(c.device), buf, NULL, NULL);
    if (err != CL_SUCCESS) {
      size_t logSize;
      clGetProgramBuildInfo(program, c.device, CL_PROGRAM_BUILD_LOG, sizeof(buf), buf, &logSize);
      buf[logSize] = 0;
      fprintf(stderr, "log %s\n", buf);
    }
    CHECK(err);

    fclose(f);
  }

 private:  
  size_t read(char *buf, int bufSize, FILE *f) {
    size_t size = fread(buf, 1, bufSize, f); assert(size);
    return size;
  }
};

    // "-Werror -cl-fast-relaxed-math -I. -fno-bin-llvmir -fno-bin-source -fno-bin-amdil -save-temps=tmp1/";
    // -cl-opt-disable

class Kernel {
public:
  cl_kernel k;
  
  Kernel(Program &p, const char *name) {
    int err;
    k = clCreateKernel(p.program, name, &err);
    CHECK2(err, name);
  }

  ~Kernel() {
    clReleaseKernel(k); k = 0;
  }

  template<class T> void setArg(int pos, const T &value) { CHECK(clSetKernelArg(k, pos, sizeof(value), &value)); }

  template<class A> void setArgs(const A &a) {
    setArg(0, a);
  }
  
  template<class A, class B> void setArgs(const A &a, const B &b) {
    setArgs(a);
    setArg(1, b);
  }
  
  template<class A, class B, class C> void setArgs(const A &a, const B &b, const C &c) {
    setArgs(a, b);
    setArg(2, c);
  }

  template<class A, class B, class C, class D> void setArgs(const A &a, const B &b, const C &c, const D &d) {
    setArgs(a, b, c);
    setArg(3, d);
  }
  
  template<class A, class B, class C, class D, class E> void setArgs(const A &a, const B &b, const C &c, const D &d, const E &e) {
    setArgs(a, b, c, d);
    setArg(4, e);
  }

  template<class A, class B, class C, class D, class E, class F> void setArgs(const A &a, const B &b, const C &c, const D &d, const E &e, const F &f) {
    setArgs(a, b, c, d, e);
    setArg(5, f);
  }
};

class Buf {
 public:
  cl_mem buf;

  Buf(Context &c, unsigned kind, size_t size, void *ptr) {
    int err;
    buf = clCreateBuffer(c.context, kind, size, ptr, &err);
    CHECK(err);
  }

  Buf(Context &c, unsigned kind, size_t size) : Buf(c, kind, size, NULL) { }

  // ~Buf() { }

  void release() { CHECK(clReleaseMemObject(buf)); }
};

struct Event {
  cl_event event;

  void release() { CHECK(clReleaseEvent(event)); }
  void wait() { CHECK(clWaitForEvents(1, &event)); }
};

class Queue {
public:
  cl_command_queue queue;
  long prevTime;

  Queue(Context &c) {
    int err;
    queue = clCreateCommandQueue(c.context, c.device, 0, &err);
    CHECK(err);
    prevTime = timeMillis();
  }

  ~Queue() {
    flush();
    CHECK(clFinish(queue));
    CHECK(clReleaseCommandQueue(queue));
  }

  long time() {
    // if (wait) { finish(); }
    long now = timeMillis();
    long delta = now - prevTime;
    prevTime = now;
    return delta;
  }
  
  void run(size_t groupSize, Kernel &k, size_t workSize) {
    CHECK(clEnqueueNDRangeKernel(queue, k.k, 1, NULL, &workSize, &groupSize, 0, NULL, NULL));
  }

  void run(Kernel &k, size_t workSize) { run(256, k, workSize); }

  template<class A> void run(Kernel &k, size_t workSize, const A &a) {
    k.setArgs(a);
    run(k, workSize);
  }

  template<class A, class B> void run(Kernel &k, size_t workSize, const A &a, const B &b) {
    k.setArgs(a, b);
    run(k, workSize);
  }

  template<class A, class B, class C, class D, class E, class F>
    void run(Kernel &k, size_t workSize,
             const A &a, const B &b, const C &c, const D &d, const E &e, const F &f) {
    k.setArgs(a, b, c, d, e, f);
    run(k, workSize);
  }

  void run2D(size_t groupSize0, size_t groupSize1, Kernel &k, size_t workSize0, size_t workSize1) {
    size_t groupSizes[] = {groupSize0, groupSize1};
    size_t workSizes[] = {workSize0, workSize1};
    CHECK(clEnqueueNDRangeKernel(queue, k.k, 2, NULL, workSizes, groupSizes, 0, NULL, NULL));
  }

  void read(bool blocking, Buf &buf, size_t size, void *data, size_t start = 0, cl_event *outEvent = NULL) {
    CHECK(clEnqueueReadBuffer(queue, buf.buf, blocking, start, size, data, 0, NULL, outEvent));
  }

  Event read(Buf &buf, size_t size, void *data, size_t start = 0) {
    Event event;
    read(false, buf, size, data, start, &event.event);
    return event;
  }
  
  void write(bool blocking, Buf &buf, size_t size, void *data, size_t start = 0) {
    CHECK(clEnqueueWriteBuffer(queue, buf.buf, blocking, start, size, data, 0, NULL, NULL));
  }


  void flush() { CHECK(clFlush(queue)); }

  void finish() { CHECK(clFinish(queue)); }
};

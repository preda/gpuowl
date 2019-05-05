// Copyright Mihai Preda.

#pragma once

#include "tinycl.h"
#include "common.h"

#include <string>
#include <vector>
#include <memory>
#include <cassert>

using cl_queue = cl_command_queue;

void release(cl_context context);
void release(cl_kernel k);
void release(cl_mem buf);
void release(cl_program program);
void release(cl_queue queue);

template<typename T>
struct Deleter {
  using pointer = T;
  void operator()(T t) const { release(t); }
};

namespace std {
template<> struct default_delete<cl_context> : public Deleter<cl_context> {};
template<> struct default_delete<cl_kernel> : public Deleter<cl_queue> {};
template<> struct default_delete<cl_mem> : public Deleter<cl_mem> {};
template<> struct default_delete<cl_program> : public Deleter<cl_queue> {};
template<> struct default_delete<cl_queue> : public Deleter<cl_queue> {};
}

template<typename T> using Holder = std::unique_ptr<T, Deleter<T> >;

using Context = std::unique_ptr<cl_context>;
using QueueHolder = std::unique_ptr<cl_queue>;

template<typename T>
class Buffer : public std::unique_ptr<cl_mem> {
  size_t _size{};
  
public:
  Buffer() = default;
  
  Buffer(cl_context context, unsigned kind, size_t size, const T* ptr = nullptr)
    : std::unique_ptr<cl_mem>{_makeBuf(context, kind, size * sizeof(T), ptr)}
    , _size(size)
  {}

  Buffer(const Context& context, unsigned kind, size_t size, const T* ptr = nullptr)
    : Buffer(context.get(), kind, size, ptr)
  {}
  
  Buffer(cl_context context, unsigned kind, const std::vector<T>& vect)
    : Buffer(context, kind, vect.size(), vect.data())
  {}

  size_t size() const { return _size; }
};

const unsigned BUF_CONST = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
const unsigned BUF_RW    = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;

void check(int err, const char *file, int line, const char *func, const std::string& mes);

#define CHECK1(err) check(err, __FILE__, __LINE__, __func__, #err)
#define CHECK2(err, mes) check(err, __FILE__, __LINE__, __func__, mes)

vector<cl_device_id> getDeviceIDs(bool onlyGPU = false);
string getHwName(cl_device_id id);
string getShortInfo(cl_device_id device);
string getLongInfo(cl_device_id device);

// Get GPU free memory in bytes.
u64 getFreeMem(cl_device_id id);

Context createContext(cl_device_id id);

cl_program compile(cl_device_id device, cl_context context, const string &source, const string &extraArgs,
                   const vector<pair<string, unsigned>> &defines);

void dumpBinary(cl_program program, const string& fileName);

cl_kernel makeKernel(cl_program program, const char *name);

template<typename T>
void setArg(cl_kernel k, int pos, const T &value) { CHECK1(clSetKernelArg(k, pos, sizeof(value), &value)); }

template<typename T>
void setArg(cl_kernel k, int pos, const Buffer<T>& buf) { setArg(k, pos, buf.get()); }

// template<> void setArg<void*>(cl_kernel k, int pos, void* const &value) {CHECK(clSetKernelArgSVMPointer(k, pos, value));}

template<>
void setArg<int>(cl_kernel k, int pos, const int &value);

// Special-case Buffer argument: pass the wrapped cl_mem.
template<typename T>
void setArg(cl_kernel k, int pos, const Buffer<T>& buf);

cl_mem _makeBuf(cl_context context, unsigned kind, size_t size, const void *ptr = 0);
// cl_mem makeBuf(Context &context, unsigned kind, size_t size, const void *ptr = 0);
cl_queue makeQueue(cl_device_id d, cl_context c);

void flush( cl_queue q);
void finish(cl_queue q);

void run(cl_queue queue, cl_kernel kernel, size_t groupSize, size_t workSize, const string &name);
void read(cl_queue queue, bool blocking, cl_mem buf, size_t size, void *data, size_t start = 0);
void write(cl_queue queue, bool blocking, cl_mem buf, size_t size, const void *data, size_t start = 0);

// template<typename T> void read(cl_queue queue, bool blocking, Buffer<T> &buf, size_t size, T *data, size_t start = 0);
// void write(cl_queue queue, bool blocking, Buffer &buf, size_t size, const void *data, size_t start = 0);

void copyBuf(cl_queue queue, const cl_mem src, cl_mem dst, size_t size);

// void copyBuf(cl_queue queue, Buffer &src, Buffer &dst, size_t size);


void fillBuf(cl_queue q, cl_mem buf, void *pat, size_t patSize, size_t size = 0, size_t start = 0);
int getKernelNumArgs(cl_kernel k);
int getWorkGroupSize(cl_kernel k, cl_device_id device, const char *name);
std::string getKernelArgName(cl_kernel k, int pos);

class Queue : public QueueHolder {
public:
  explicit Queue(cl_queue queue) : QueueHolder(queue) {}

  template<typename T> vector<T> read(const Buffer<T>& buf, size_t sizeOrFull = 0) {
    auto size = sizeOrFull ? sizeOrFull : buf.size();
    assert(size <= buf.size());
    vector<T> ret(size);
    ::read(get(), true, buf.get(), size * sizeof(T), ret.data());
    return ret;
  }

  template<typename T> void readAsync(const Buffer<T>& buf, vector<T>& out, size_t sizeOrFull = 0) {
    auto size = sizeOrFull ? sizeOrFull : buf.size();
    assert(size <= buf.size());
    out.resize(size);
    ::read(get(), false, buf.get(), size * sizeof(T), out.data());
  }
    
  template<typename T> void write(Buffer<T>& buf, const vector<T> &vect) {
    assert(vect.size() <= buf.size());
    ::write(get(), true, buf.get(), vect.size() * sizeof(T), vect.data());
  }

  template<typename T> void writeAsync(Buffer<T>& buf, const vector<T> &vect) {
    assert(vect.size() <= buf.size());
    ::write(get(), false, buf.get(), vect.size() * sizeof(T), vect.data());
  }

  template<typename T> void copyFromTo(const Buffer<T>& src, Buffer<T>& dst) {
    assert(src.size() <= dst.size());
    copyBuf(get(), src.get(), dst.get(), src.size() * sizeof(T));
  }
  
  void run(cl_kernel kernel, size_t groupSize, size_t workSize, const string &name) {
    ::run(get(), kernel, groupSize, workSize, name);
  }

  void finish() { ::finish(get()); }

  template<typename T> void zero(Buffer<T>& buf, size_t sizeOrFull = 0) {
    auto size = sizeOrFull ? sizeOrFull : buf.size();
    assert(size <= buf.size());
    T zero = 0;
    fillBuf(get(), buf.get(), &zero, sizeof(T), size * sizeof(T));
  }
};

cl_device_id getDevice(int argsDevId);

// How many blocks of given size can be allocated on the device.
u32 getAllocableBlocks(cl_device_id device, u32 blockSize, u32 minFree = 400 * 1024 * 1024);

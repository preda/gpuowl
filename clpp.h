// Copyright Mihai Preda.

#pragma once

#include "clwrap.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <memory>

template<typename T> class Buffer;

class Context : public std::unique_ptr<cl_context> {
  static constexpr unsigned BUF_CONST = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
  static constexpr unsigned BUF_RW    = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;

public:
  using std::unique_ptr<cl_context>::unique_ptr;
  
  template<typename T> auto constBuf(const std::vector<T>& vect) const { return Buffer{*this, BUF_CONST, vect}; }
  template<typename T> auto hostAccessBuf(size_t size) const { return Buffer<T>{*this, CL_MEM_READ_WRITE, size}; }
  template<typename T> auto buffer(size_t size) const { return Buffer<T>{*this, BUF_RW, size}; }
};

template<typename T>
class Buffer : public std::unique_ptr<cl_mem> {
  size_t _size{};
  std::string name;

  Buffer(cl_context context, unsigned kind, size_t size, const T* ptr = nullptr)
    : std::unique_ptr<cl_mem>{_makeBuf(context, kind, size * sizeof(T), ptr)}
    , _size(size)
  {}
    
public:
  using type = T;
  
  Buffer() = default;

  Buffer(const Context& context, unsigned kind, size_t size, const T* ptr = nullptr)
    : Buffer(context.get(), kind, size, ptr)
  {}
  
  Buffer(const Context& context, unsigned kind, const std::vector<T>& vect)
    : Buffer(context.get(), kind, vect.size(), vect.data())
  {}

  Buffer(Buffer&& rhs) = default;
  Buffer& operator=(Buffer&& rhs) = default;
  
  size_t size() const { return _size; }
};

// Special-case Buffer argument: pass the wrapped cl_mem.
template<typename T>
void setArg(cl_kernel k, int pos, const Buffer<T>& buf) { setArg(k, pos, buf.get()); }


class Queue {
  std::shared_ptr<QueueHolder> q;
  
public:
  explicit Queue(cl_queue q) : q(make_shared<QueueHolder>(q)) {}

  cl_queue get() const { return q->get(); }

  template<typename T> vector<T> read(const Buffer<T>& buf, size_t sizeOrFull = 0) {
    auto size = sizeOrFull ? sizeOrFull : buf.size();
    assert(size <= buf.size());
    vector<T> ret(size);
    ::read(q->get(), true, buf.get(), size * sizeof(T), ret.data());
    return ret;
  }

  template<typename T> void readAsync(const Buffer<T>& buf, vector<T>& out, size_t sizeOrFull = 0) {
    auto size = sizeOrFull ? sizeOrFull : buf.size();
    assert(size <= buf.size());
    out.resize(size);
    ::read(q->get(), false, buf.get(), size * sizeof(T), out.data());
  }
    
  template<typename T> void write(Buffer<T>& buf, const vector<T> &vect) {
    assert(vect.size() <= buf.size());
    ::write(q->get(), true, buf.get(), vect.size() * sizeof(T), vect.data());
  }

  template<typename T> void writeAsync(Buffer<T>& buf, const vector<T> &vect) {
    assert(vect.size() <= buf.size());
    ::write(q->get(), false, buf.get(), vect.size() * sizeof(T), vect.data());
  }

  template<typename T> void copyFromTo(const Buffer<T>& src, Buffer<T>& dst) {
    assert(src.size() <= dst.size());
    copyBuf(q->get(), src.get(), dst.get(), src.size() * sizeof(T));
  }
  
  void run(cl_kernel kernel, size_t groupSize, size_t workSize, const string &name) {
    ::run(q->get(), kernel, groupSize, workSize, name);
  }

  void finish() { ::finish(q->get()); }

  template<typename T> void zero(Buffer<T>& buf, size_t sizeOrFull = 0) {
    auto size = sizeOrFull ? sizeOrFull : buf.size();
    assert(size <= buf.size());
    T zero = 0;
    fillBuf(q->get(), buf.get(), &zero, sizeof(T), size * sizeof(T));
  }
};

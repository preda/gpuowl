// Copyright Mihai Preda.

#pragma once

#include "clwrap.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

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

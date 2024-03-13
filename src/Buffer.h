// Copyright (C) Mihai Preda.

#pragma once

#include "clwrap.h"
#include "AllocTrac.h"
#include "Context.h"
#include "Queue.h"

#include <memory>
#include <string>
#include <vector>
#include <cassert>

class TimeInfo;

template<typename T>
class Buffer {
private:
  std::unique_ptr<cl_mem> ptr;

public:
  const size_t size{};

private:
  const std::string name;
  AllocTrac allocTrac;

  QueuePtr queue;
  TimeInfo *tInfo;
  
  Buffer(TimeInfo *tInfo, QueuePtr queue, std::string_view name, size_t size, unsigned flags, const T* ptr = nullptr)
    : ptr{makeBuf_(getQueueContext(queue->get()), flags, size * sizeof(T), ptr)}
    , size{size}
    , name{name}
    , allocTrac(size * sizeof(T))
    , queue{queue}
    , tInfo{tInfo}
  {}

  void fill(T value, u32 len) {
    assert(len <= size);
    queue->fillBuf(get(), value, (len ? len : size) * sizeof(T), tInfo);
  }

public:
  Buffer(TimeInfo* tInfo, QueuePtr queue, std::string_view name, const std::vector<T>& vect)
    : Buffer(tInfo, queue, name,
             vect.size(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, vect.data())
  {}


  Buffer(TimeInfo *tInfo, QueuePtr queue, std::string_view name, size_t size)
    : Buffer(tInfo, queue, name, size, CL_MEM_READ_WRITE /*| CL_MEM_HOST_NO_ACCESS*/) {}

  Buffer(Buffer&& rhs) = default;

  Buffer& operator=(Buffer&& rhs) {
    assert(size == rhs.size);
    std::swap(ptr, rhs.ptr);
    return *this;
  }

  cl_mem get() const { return ptr.get(); }

  // sync read
  vector<T> read(size_t sizeOrFull = 0) const {
    auto readSize = sizeOrFull ? sizeOrFull : size;
    assert(readSize <= size);
    vector<T> ret(readSize);
    queue->readSync(get(), readSize * sizeof(T), ret.data(), tInfo);
    return ret;
  }

  void readAsync(vector<T>& out, size_t sizeOrFull = 0) const {
    auto readSize = sizeOrFull ? sizeOrFull : size;
    assert(readSize <= size);
    out.resize(readSize);
    queue->readAsync(get(), readSize * sizeof(T), out.data(), tInfo);
  }

  void write(vector<i32>&& vect) { queue->writeAsync(get(), std::move(vect), tInfo); }

#if 0
  void write(const vector<T>& vect) {
    assert(size >= vect.size());
    queue->write(get(), vect.size() * sizeof(T), vect.data());
  }
#endif


  void zero(size_t len = 0) {
    fill(0, len);
  }

  void set(T value) {
    zero();
    fill(value, 1);
  }

  // device-side copy
  void operator<<(const Buffer<T>& rhs) {
    assert(size == rhs.size);
    queue->copyBuf(rhs.get(), get(), size * sizeof(T), tInfo);
    // copyBuf(queue->get(), rhs.get(), get(), size * sizeof(T));
  }
};

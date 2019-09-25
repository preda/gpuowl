// Copyright Mihai Preda.

#pragma once

#include "clwrap.h"
#include "AllocTrac.h"
#include "Context.h"
#include "Queue.h"

#include <memory>
#include <string>
#include <vector>

template<typename T>
class ConstBuffer {
  std::unique_ptr<cl_mem> ptr;
  
public:
  const size_t size{};
  const std::string name;

private:
  AllocTrac allocTrac;

protected:
  ConstBuffer(cl_context context, std::string_view name, unsigned kind, size_t size, const T* ptr = nullptr)
    : ptr{makeBuf_(context, kind, size * sizeof(T), ptr)}
    , size(size)
    , name(name)
    , allocTrac(size * sizeof(T))
  {}
    
public:
  using type = T;
  
  ConstBuffer() = delete;

  ConstBuffer(const Context& context, std::string_view name, unsigned kind, size_t size, const T* ptr = nullptr)
    : ConstBuffer(context.get(), name, kind, size, ptr)
  {}

  ConstBuffer(const Context& context, std::string_view name, const std::vector<T>& vect)
    : ConstBuffer(context.get(), name, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, vect.size(), vect.data())
  {}
  
  ConstBuffer(ConstBuffer&& rhs) = default;
  
  ConstBuffer& operator=(ConstBuffer&& rhs) {
    assert(size == rhs.size);
    ptr = std::move(rhs.ptr);
    return *this;
  }
  
  virtual ~ConstBuffer() = default;
  
  cl_mem get() const { return ptr.get(); }
  void reset() { ptr.reset(); }
};

template<typename T>
class Buffer : public ConstBuffer<T> {
protected:
  QueuePtr queue;
  
  Buffer(QueuePtr queue, std::string_view name, size_t size, unsigned kind)
    : ConstBuffer<T>{getQueueContext(queue->get()), name, kind, size}
    , queue{queue}
  {}
    
public:
  Buffer(QueuePtr queue, std::string_view name, size_t size)
    : Buffer(queue, name, size, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS) {}

  Buffer(Buffer&& rhs) = default;
  
  void operator=(const ConstBuffer<T>& rhs) {
    assert(this->size == rhs.size);
    copyBuf(queue->get(), rhs.get(), this->get(), this->size * sizeof(T));
  }

  void operator=(const Buffer<T>& rhs) { *this = static_cast<const ConstBuffer<T>&>(rhs); }

  void operator=(const vector<T>& vect) {
    assert(size == vect.size());
    write(queue->get(), true, this->get(), vect.size() * sizeof(T), vect.data());
  }
};

template<typename T>
class HostAccessBuffer : public Buffer<T> {
public:
  using Buffer<T>::operator=;
  
  HostAccessBuffer(QueuePtr queue, std::string_view name, size_t size)
    : Buffer<T>(queue, name, size, CL_MEM_READ_WRITE) {}

  operator vector<T>() const {
    vector<T> ret(size);
    read(this->queue->get(), true, this->get(), this->size * sizeof(T), ret.data());
    return ret;
  }
};

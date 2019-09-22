// Copyright Mihai Preda.

#pragma once

#include "clwrap.h"
#include "AllocTrac.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

template<typename T> class Buffer;

class Context : public std::unique_ptr<cl_context> {
  static constexpr unsigned BUF_CONST = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;

  cl_device_id id;
public:
  Context(cl_device_id id): unique_ptr<cl_context>{createContext(id)}, id{id} {}
  
  cl_device_id deviceId() const { return id; }
  
  template<typename T>
  auto constBuf(const std::vector<T>& vect, std::string_view name) const {
    return Buffer{*this, name, BUF_CONST, vect.size(), vect.data()};
  }
  
  template<typename T>
  auto hostAccessBuf(size_t size, std::string_view name) const {
    return Buffer<T>{*this, name, CL_MEM_READ_WRITE, size};
  }
  
  template<typename T>
  auto buffer(size_t size, std::string_view name) const {
    return Buffer<T>{*this, name, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, size};
  }
};

template<typename T>
class Buffer : public std::unique_ptr<cl_mem> {
  size_t size_{};
  std::string name_;
  AllocTrac allocTrac;

  Buffer(cl_context context, std::string_view name, unsigned kind, size_t size, const T* ptr = nullptr)
    : std::unique_ptr<cl_mem>{makeBuf_(context, kind, size * sizeof(T), ptr)}
    , size_(size)
    , name_(name)
    , allocTrac(size * sizeof(T))
  {}
    
public:
  using type = T;
  
  Buffer() = default;

  Buffer(const Context& context, std::string_view name, unsigned kind, size_t size, const T* ptr = nullptr)
    : Buffer(context.get(), name, kind, size, ptr)
  {}

  size_t size() const { return size_; }
  const std::string& name() const { return name_; }
};

// Special-case Buffer argument: pass the wrapped cl_mem.
template<typename T>
void setArg(cl_kernel k, int pos, const Buffer<T>& buf) { setArg(k, pos, buf.get()); }

class Event : public EventHolder {
public:
  double secs() { return float(getEventNanos(this->get())) * 1e-9f; }
  bool isComplete() { return getEventInfo(this->get()) == CL_COMPLETE; }
};

// Copyright Mihai Preda.

#pragma once

#include "clwrap.h"

#include <memory>
#include <string>
#include <vector>

template<typename T> class ConstBuffer;
template<typename T> class Buffer;

class Context : public std::unique_ptr<cl_context> {
  cl_device_id id;
public:
  Context(cl_device_id id): unique_ptr<cl_context>{createContext(id)}, id{id} {}
  
  cl_device_id deviceId() const { return id; }

  /*
  template<typename T>
  auto hostAccessBuf(size_t size, std::string_view name) const {
    return Buffer<T>{*this, name, CL_MEM_READ_WRITE, size};
  }
  
  template<typename T>
  auto buffer(size_t size, std::string_view name) const {
    return ConstBuffer<T>{*this, name, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, size};
  }
  */
};

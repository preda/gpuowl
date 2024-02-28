// Copyright Mihai Preda.

#pragma once

#include "clwrap.h"

#include <memory>

class Context : public std::unique_ptr<cl_context> {
  cl_device_id id;
public:
  explicit Context(cl_device_id id): unique_ptr<cl_context>{createContext(id)}, id{id} {}
  
  cl_device_id deviceId() const { return id; }
};

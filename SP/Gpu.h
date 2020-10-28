// Copyright Mihai Preda.

#pragma once

#include "Buffer.h"
#include "Context.h"
#include "Queue.h"

#include "common.h"
#include "kernel.h"

#include <vector>
#include <string>
#include <memory>
#include <variant>
#include <atomic>
#include <future>
#include <filesystem>

using double2 = pair<double, double>;
using float2 = pair<float, float>;

class Gpu {
  u32 ND;
  cl_device_id device;
  Context context;
  QueuePtr queue;
  // Holder<cl_program> program;  
  // optional<Kernel> readHwTrig;

  string readTrigTable(); 
  
public:
  void finish() { queue->finish(); }

  Gpu();
};

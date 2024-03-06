// Copyright (C) Mihai Preda

#include "Kernel.h"
#include "KernelCompiler.h"

#include <stdexcept>

Kernel::Kernel(string_view name, QueuePtr queue,
       string_view fileName, string_view nameInFile,
       size_t workSize, string_view defines):
  name{name},
  fileName{fileName},
  nameInFile{nameInFile},
  defines{defines},
  queue{queue},
  workSize{workSize}
{}

void Kernel::load(const KernelCompiler& compiler, cl_device_id deviceId) {
  assert(!kernel);
  kernel = compiler.load(fileName, nameInFile, defines);
  assert(kernel);
  groupSize = getWorkGroupSize(kernel.get(), deviceId, name.c_str());
  assert(groupSize);
  assert(workSize % groupSize == 0);
}

void Kernel::run() {
  if (kernel) {
    queue->run(kernel.get(), groupSize, workSize, name);
  } else {
    throw std::runtime_error("OpenCL kernel "s + name + " not found");
  }
}

// Copyright (C) Mihai Preda

#include "Kernel.h"
#include "KernelCompiler.h"

#include <stdexcept>

Kernel::Kernel(string_view name, TimeInfo* timeInfo, Queue* queue,
       string_view fileName, string_view nameInFile,
       size_t workSize, string_view defines):
  name{name},
  fileName{fileName},
  nameInFile{nameInFile},
  defines{defines},
  timeInfo{timeInfo},
  queue{queue},
  workSize{workSize}
{}

Kernel::~Kernel() = default;

void Kernel::startLoad(const KernelCompiler& compiler) {
  assert(!kernel);
  assert(!pendingKernel.valid());
  pendingKernel = compiler.load(fileName, nameInFile, defines);
  deviceId = compiler.deviceId;
}

void Kernel::finishLoad() {
  pendingKernel.wait();
  kernel = pendingKernel.get();
  assert(kernel);
  groupSize = getWorkGroupSize(kernel.get(), deviceId, name.c_str());
  assert(groupSize);
  assert(workSize % groupSize == 0);

  for (auto [pos, arg] : pendingArgs) { setArgs(pos, arg); }
}

void Kernel::run() {
  if (kernel) {
    queue->run(kernel.get(), groupSize, workSize, timeInfo);
  } else {
    throw std::runtime_error("OpenCL kernel "s + name + " not found");
  }
}

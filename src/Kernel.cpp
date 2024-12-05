// Copyright (C) Mihai Preda

#include "Kernel.h"
#include "KernelCompiler.h"

#include <stdexcept>

Kernel::Kernel(string_view name, KernelCompiler* compiler, TimeInfo* timeInfo, Queue* queue,
       string_view fileName, string_view nameInFile,
       size_t workSize, u32 nGroups, string_view defines):
  name{name},
  compiler{compiler},
  fileName{fileName},
  nameInFile{nameInFile},
  defines{defines},
  timeInfo{timeInfo},
  queue{queue},
  workSize{workSize},
  nGroups{nGroups}
{
  assert(!workSize || !nGroups); // Use only one way to indicate size: either threads or groups.
}

Kernel::~Kernel() = default;

void Kernel::startLoad(KernelCompiler* compiler) {
  assert(!kernel);
  assert(!pendingKernel.valid());
  pendingKernel = compiler->load(fileName, nameInFile, defines);
  deviceId = compiler->deviceId;
}

void Kernel::finishLoad() {
  pendingKernel.wait();
  kernel = pendingKernel.get();
  assert(kernel);
  groupSize = getWorkGroupSize(kernel.get(), deviceId, name.c_str());
  assert(groupSize);
  if (!workSize) {
    assert(nGroups);
    workSize = groupSize * nGroups;
  } else {
    assert(workSize % groupSize == 0);
    assert(nGroups == 0);
    nGroups = workSize / groupSize;
  }

  for (auto [pos, arg] : pendingArgs) { setArgs(pos, arg); }
}

void Kernel::run() {
  assert(kernel);
  queue->run(kernel.get(), groupSize, workSize, timeInfo);
}

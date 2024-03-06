#include "kernel.h"
#include "KernelCompiler.h"

void Kernel::load(const KernelCompiler& compiler, cl_device_id deviceId) {
  assert(!kernel);
  kernel = compiler.load(fileName, nameInFile, defines);
  assert(kernel);
  groupSize = getWorkGroupSize(kernel.get(), deviceId, name.c_str());
  assert(groupSize);
  assert(workSize % groupSize == 0);
}

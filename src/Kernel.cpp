#include "kernel.h"
#include "KernelCompiler.h"

void Kernel::load(const KernelCompiler& compiler, cl_device_id deviceId) {
  assert(!kernel);
  kernel = compiler.load(fileName, nameInFile, defines);
  assert(kernel);
  [[maybe_unused]] u32 gs = getWorkGroupSize(kernel.get(), deviceId, name.c_str());
  assert(gs == groupSize);
}

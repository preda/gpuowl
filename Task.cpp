#include "Task.h"

#include "Gpu.h"
#include "Result.h"
#include "checkpoint.h"
#include "args.h"

#include <cstdio>
#include <cmath>

bool Task::execute(const Args &args) {
  assert(kind == PRP);
  auto gpu = Gpu::make(exponent, args);
  return gpu->isPrimePRP(exponent, args).write(args, *this, gpu->getFFTSize());
}

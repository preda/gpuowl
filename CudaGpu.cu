#include "CudaGpu.h"

const char *VARIANT = "CUDA";

unique_ptr<Gpu> makeGpu(u32 E, Args &args) { return CudaGpu::make(E, args); }

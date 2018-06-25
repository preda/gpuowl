#include "OpenGpu.h"

const char *VARIANT = "OpenCL";

unique_ptr<Gpu> makeGpu(u32 E, Args &args) { return OpenGpu::make(E, args); }

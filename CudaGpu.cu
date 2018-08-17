#include "CudaGpu.h"
#include "TF.h"

const char *VARIANT = "CUDA";

unique_ptr<Gpu> makeGpu(u32 E, Args &args) { return CudaGpu::make(E, args); }

bool TF::enabled() { return false; }

unique_ptr<TF> makeTF(Args &args) { return 0; }

vector<string> getDevices() {
  return vector<string>(); // TODO: implement.
}

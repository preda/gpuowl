#include "OpenGpu.h"
#include "TF.h"

const char *VARIANT = "OpenCL";

unique_ptr<Gpu> makeGpu(u32 E, Args &args) { return OpenGpu::make(E, args); }

unique_ptr<TF> makeTF(Args &args) { return unique_ptr<TF>(); }

vector<string> getDevices() {
  vector<string> ret;
  for (auto id : getDeviceIDs(false)) { ret.push_back(getLongInfo(id)); }
  return ret;
}


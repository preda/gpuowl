#include "OpenGpu.h"
#include "OpenTF.h"

const char *VARIANT = "OpenCL";

unique_ptr<Gpu> makeGpu(u32 E, Args &args) { return OpenGpu::make(E, args); }

bool TF::enabled() { return true; }

unique_ptr<TF> makeTF(Args &args) { return OpenTF::make(args); }

vector<string> getDevices() {
  vector<string> ret;
  for (auto id : getDeviceIDs(false)) { ret.push_back(getLongInfo(id)); }
  return ret;
}

#include "OpenGpu.h"

unique_ptr<Gpu> makeGpu(u32 E, Args &args) { return OpenGpu::make(E, args); }

vector<string> getDevices() {
  vector<string> ret;
  for (auto id : getDeviceIDs(false)) { ret.push_back(getLongInfo(id)); }
  return ret;
}

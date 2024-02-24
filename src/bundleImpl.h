#include "bundle.h"

std::map<std::string, std::string> getCLSources() {
  int n = sizeof(CL_SOURCES) / sizeof(CL_SOURCES[0]);
  std::map<std::string, std::string> ret{};
  
  for (int i = 0; i < n; ++i) {
    ret[CL_FILES[i]] = CL_SOURCES[i];
  }
    
  return ret;
}

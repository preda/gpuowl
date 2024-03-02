#include "KernelCompiler.h"
#include "log.h"
#include "timeutil.h"
#include <cassert>

using namespace std;

// Implemented in bundle.cpp
const std::vector<const char*>& getClFileNames();
const std::vector<const char*>& getClFiles();

static_assert(sizeof(Program) == sizeof(cl_program));

Program KernelCompiler::newProgram(const string& fileName) const {
  for (const auto& [name, src] : files) {
    if (name == fileName) {
      return loadSource(context, src);
    }
  }
  log("KernelCompiler: can't find '%s'\n", fileName.c_str());
  return {};
}

KernelCompiler::KernelCompiler(cl_context context, cl_device_id deviceId, const string& args) :
  context{context},
  deviceId{deviceId},
  linkArgs{"-cl-finite-math-only " },
  baseArgs{linkArgs + "-cl-std=CL2.0 " + args}
{
  auto& clNames = getClFileNames();
  auto& clFiles = getClFiles();
  assert(clNames.size() == clFiles.size());
  int n = clNames.size();
  for (int i = 0; i < n; ++i) {
    auto &src = clFiles[i];
    files.push_back({clNames[i], src});
    clSources.push_back(loadSource(context, src));
  }
  log("CL args: %s\n", baseArgs.c_str());
}

Program KernelCompiler::compile(const string& fileName, const string& extraArgs) const {
  Program p1 = newProgram(fileName);
  if (!p1) { return {}; }
  
  string args = baseArgs + ' ' + extraArgs;
  int err = clCompileProgram(p1.get(), 1, &deviceId, args.c_str(),
                             clSources.size(), (const cl_program*) (clSources.data()), getClFileNames().data(),
                             nullptr, nullptr);
  if (string mes = getBuildLog(p1.get(), deviceId); !mes.empty()) { log("%s\n", mes.c_str()); }
  if (err != CL_SUCCESS) {
    log("Compiling '%s' error %d (args %s)\n", fileName.c_str(), err, args.c_str());
    return {};
  }
  
  Program p2{clLinkProgram(context, 1, &deviceId, linkArgs.c_str(),
                           1, (cl_program *) &p1, nullptr, nullptr, &err)};
  if (string mes = getBuildLog(p1.get(), deviceId); !mes.empty()) { log("%s\n", mes.c_str()); }
  if (err != CL_SUCCESS) {
    log("OpenCL link error %s (%d) (args %s)\n", errMes(err).c_str(), err, linkArgs.c_str());
  }
  return p2;
}

KernelHolder KernelCompiler::load(const string& fileName, const string& kernelName, const string& args) const {
  Timer timer;
  Program program = compile(fileName, args);
  KernelHolder ret{makeKernel(program.get(), kernelName.c_str())};
  if (!ret) {
    log("Can't load kernel '%s' from '%s'\n", kernelName.c_str(), fileName.c_str());
    throw "Can't load "s + kernelName;
  }
  log("Loaded kernel '%s' from '%s' in %.0fms\n", kernelName.c_str(), fileName.c_str(), timer.at() * 1000);
  return ret;
}

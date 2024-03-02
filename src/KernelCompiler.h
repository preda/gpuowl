// Copyright (C) Mihai Preda

#pragma once

#include "clwrap.h"

#include <vector>
#include <string>

class KernelCompiler {
  cl_context context;
  cl_device_id deviceId;
  std::string linkArgs;
  std::string baseArgs;
  
  std::vector<Program> clSources;
  std::vector<std::pair<std::string, std::string>> files;
  
  Program newProgram(const string& fileName) const;
  Program compile(const string& fileName, const string& args) const;
  
public:
  KernelCompiler(cl_context context, cl_device_id deviceId, const string& args);
  
  KernelHolder load(const string& fileName, const string& kernelName, const string& args) const;
};

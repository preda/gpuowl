// Copyright (C) Mihai Preda

#pragma once

#include "clwrap.h"

#include "common.h"
#include <vector>

class KernelCompiler {
  cl_context context;
  cl_device_id deviceId;
  
  std::vector<Program> clSources;
  std::vector<std::pair<std::string, std::string>> files;
  
  Program newProgram(const string& fileName);
  Program compile(const string& fileName, const string& args);
  
public:
  KernelCompiler(cl_context context, cl_device_id deviceId);    
};

// Copyright (C) Mihai Preda

#pragma once

#include "clwrap.h"

#include <vector>
#include <string>

class Args;

class KernelCompiler {
  std::string cacheDir;
  cl_context context;
  cl_device_id deviceId;
  std::string linkArgs;
  std::string baseArgs;
  std::string dump;
  const bool useCache;
  const bool verbose;
  
  std::vector<Program> clSources;
  std::vector<std::pair<std::string, std::string>> files;

  u64 contextHash{};
  
  Program compile(const string& fileName, const string& args) const;
  
public:
  KernelCompiler(const Args& args, cl_context context, cl_device_id deviceId, const string& clArgs);
  
  KernelHolder load(const string& fileName, const string& kernelName, const string& args) const;
};

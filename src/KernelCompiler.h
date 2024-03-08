// Copyright (C) Mihai Preda

#pragma once

#include "clwrap.h"

#include <vector>
#include <string>

class KernelCompiler {
  std::string cacheDir;
  cl_context context;
  cl_device_id deviceId;
  std::string linkArgs;
  std::string baseArgs;
  std::string dump;
  const bool useCache{false};
  
  std::vector<Program> clSources;
  std::vector<std::pair<std::string, std::string>> files;

  u64 contextHash{};
  
  Program compile(const string& fileName, const string& args) const;
  
public:
  KernelCompiler(string_view cacheDir, cl_context context, cl_device_id deviceId,
                 const string& args, string_view dump);
  
  KernelHolder load(const string& fileName, const string& kernelName, const string& args) const;
};

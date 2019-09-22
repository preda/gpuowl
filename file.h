// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include <cstdio>
#include <memory>

namespace std {
  template<> struct default_delete<FILE> {
    void operator()(FILE *f) { if (f != nullptr) { fclose(f); } }
  };
}

std::unique_ptr<FILE> openRead(const std::string &name, bool logError = false);
std::unique_ptr<FILE> openWrite(const std::string &name);
std::unique_ptr<FILE> openAppend(const std::string &name);

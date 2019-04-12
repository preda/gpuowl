// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include <cstdio>
#include <memory>

namespace std {
  template<> struct default_delete<FILE> {
    void operator()(FILE *f) { if (f != nullptr) { fclose(f); } }
  };
}

unique_ptr<FILE> openRead(const string &name, bool logError = false);
unique_ptr<FILE> openWrite(const string &name);
unique_ptr<FILE> openAppend(const string &name);

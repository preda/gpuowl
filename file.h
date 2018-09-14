#pragma once

#include <cstdio>
#include <memory>

namespace std {
  template<> struct default_delete<FILE> {
    void operator()(FILE *f) { if (f != nullptr) { fclose(f); } }
  };
}

unique_ptr<FILE> open(const string &name, const char *mode, bool doLog = true);

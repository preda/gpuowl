// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "Kset.h"

#include "file.h"
#include <cassert>
#include <algorithm>

Kset::Kset(const string &name) {
  if (!name.empty()) {
    if (auto fi = openRead(name, true)) {
      u32 k;
      while (fscanf(fi.get(), "%u", &k) == 1) { Ks.push_back(k); }

    }
  }
  Ks.push_back(0xffffffffu); // guard
  hint = Ks.begin();
}

u32 Kset::getFirstAfter(u32 k) {
  assert(hint != Ks.end());
  if (k < *hint) { hint = Ks.begin(); }
  hint = upper_bound(hint, Ks.end(), k);
  assert(hint != Ks.end());
  return *hint;
}

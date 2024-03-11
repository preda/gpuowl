// Copyright (C) Mihai Preda

#pragma once

#include "common.h"

#include <array>

class TimeInfo {
public:
  const std::string name;

  explicit TimeInfo(string_view s);
  ~TimeInfo();


  std::array<i64, 3> times{};
  u32 n{};

  void add(std::array<i64, 3> ts) {
    for (int i = 0; i < 3; ++i) { times[i] += ts[i]; }
    ++n;
  }

  void clear() {
    for (int i = 0; i < 3; ++i) { times[i] = 0; }
    n = 0;
  }

  bool operator<(const TimeInfo& rhs) const { return times[2] > rhs.times[2]; }

  auto secs() const {
    std::array<double, 3> ret{};
    for (int i = 0; i < 3; ++i) { ret[i] = times[i] * 1e-9; }
    return ret;
  }
};

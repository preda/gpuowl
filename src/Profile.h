// Copyright (C) Mihai Preda

#pragma once

#include "TimeInfo.h"

#include <memory>
#include <string_view>
#include <vector>

class TimeInfo;

class Profile {
  std::vector<std::unique_ptr<TimeInfo>> entries;

public:
  TimeInfo *make(std::string_view s);

  std::vector<const TimeInfo*> get() const;

  void reset();
};

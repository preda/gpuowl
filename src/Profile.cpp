// Copyright (C) Mihai Preda

#include "Profile.h"
#include "TimeInfo.h"

#include <algorithm>

TimeInfo* Profile::make(string_view s) {
  entries.push_back(make_unique<TimeInfo>(s));
  return entries.back().get();
}

vector<const TimeInfo*> Profile::get() const {
  vector<const TimeInfo*> ret;
  for (auto& t : entries) { if (t->n) { ret.push_back(t.get()); } }
  std::sort(ret.begin(), ret.end(), [](auto p1, auto p2) { return p1->times[2] > p2->times[2]; });
  return ret;
}

void Profile::reset() {
  for (auto& t : entries) { t->clear(); }
}

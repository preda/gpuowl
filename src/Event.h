// Copyright (C) Mihai Preda

#pragma once

#include "clwrap.h"

class TimeInfo;

class Event {
  // mutable bool isFinalized{false};

public:
  EventHolder event;
  TimeInfo *tInfo;
  u64 seq;

  Event(EventHolder&& e, TimeInfo *tInfo, u64 seq);
  Event(Event&& oth) = default;
  ~Event();

  cl_event get() const { return event.get(); }

  bool isComplete();
  bool isRunning();
  bool isQueued();
  bool isSubmitted();
};

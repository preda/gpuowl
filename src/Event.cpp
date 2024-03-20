// Copyright (C) Mihai Preda

#include "Event.h"
#include "TimeInfo.h"

#include <cassert>

Event::Event(EventHolder&& e, TimeInfo* tInfo, u64 seq) :
  event{std::move(e)},
  tInfo{tInfo},
  seq{seq}
{
  assert(tInfo);
}

Event::~Event() {
  [[maybe_unused]] bool done = isComplete();
  assert(done);
}

bool Event::isComplete() {
  if (event && getEventInfo(event.get()) == CL_COMPLETE) {
      tInfo->add(getEventNanos(get()));
      event.reset();
  }
  return !event;
}

bool Event::isRunning() { return event && getEventInfo(event.get()) == CL_RUNNING; }
bool Event::isQueued() { return event && getEventInfo(event.get()) == CL_QUEUED; }
bool Event::isSubmitted() { return event && getEventInfo(event.get()) == CL_SUBMITTED; }

// Copyright (C) Mihai Preda

#include "Event.h"
#include "TimeInfo.h"

#include <cassert>

Event::Event(EventHolder&& e, TimeInfo* tInfo) :
  event{std::move(e)},
  tInfo{tInfo}
{
  assert(tInfo);
}

Event::~Event() {
  if (event) {
    [[maybe_unused]] bool finalized = isDone();
    assert(finalized);
  }
}

bool Event::isDone() const {
  if (!isFinalized) {
    if (getEventInfo(event.get()) == CL_COMPLETE) {
      tInfo->add(getEventNanos(get()));
      isFinalized = true;
    }
  }
  return isFinalized;
}

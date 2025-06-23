// Copyright (C) Mihai Preda

#include "Queue.h"
#include "Args.h"
#include "TimeInfo.h"
#include "timeutil.h"
#include "log.h"

#include <cassert>

void Events::clearCompleted() { while (!empty() && front().isComplete()) { pop_front(); } }

void Events::synced() {
  clearCompleted();
  assert(empty());
}

Queue::Queue(const Context& context, bool profile) :
  QueueHolder{makeQueue(context.deviceId(), context.get(), profile)},
  hasEvents{profile},
  context{&context},
  markerEvent{},
  markerQueued(false),
  queueCount(0),
  squareTime(50)
{
  // Formerly a constant (thus the CAPS).  nVidia is 3% CPU load at 400 or 500, and 35% load at 800 on my Linux machine.
  // AMD is just over 2% load at 1600 and 3200 on the same Linux machine.  Marginally better timings(?) at 3200.
  MAX_QUEUE_COUNT = isAmdGpu(context.deviceId()) ? 3200 : 500;		// Queue size for 800 or 125 squarings
}

void Queue::writeTE(cl_mem buf, u64 size, const void* data, TimeInfo* tInfo) {
  add(::write(get(), {}, true, buf, size, data, hasEvents), tInfo);
  events.synced();
}

void Queue::fillBufTE(cl_mem buf, u32 patSize, const void* pattern, u64 size, TimeInfo* tInfo) {
  add(::fillBuf(get(), {}, buf, pattern, patSize, size, hasEvents), tInfo);
}

string status(Events& events) {
  if (events.empty()) { return ""; }
  Event& f = events.front();
  return f.isComplete() ? "C" : f.isQueued() ? "Q" : f.isRunning() ? "R" : f.isSubmitted() ? "S" : "?";
}

void Queue::print() {
  events.clearCompleted();
  log("%s\n", status(events).c_str());
}

void Queue::add(EventHolder&& e, TimeInfo* ti) {
  if (hasEvents) { events.emplace_back(std::move(e), ti); }
  queueCount++;
  if (queueCount == MAX_QUEUE_COUNT) queueMarkerEvent();
}

void Queue::readSync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo) {
  queueMarkerEvent();
  add(read(get(), {}, true, buf, size, out, hasEvents), tInfo);
  events.synced();
}

void Queue::readAsync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo) {
  add(read(get(), {}, false, buf, size, out, hasEvents), tInfo);
}

void Queue::copyBuf(cl_mem src, cl_mem dst, u32 size, TimeInfo* tInfo) {
  add(::copyBuf(get(), {}, src, dst, size, hasEvents), tInfo);
}

void Queue::run(cl_kernel kernel, size_t groupSize, size_t workSize, TimeInfo* tInfo) {
  add(::run(get(), kernel, groupSize, workSize, {}, tInfo->name, hasEvents), tInfo);
}

void Queue::finish() {
  waitForMarkerEvent();
  ::finish(get());
  events.synced();
  queueCount = 0;
}

void Queue::queueMarkerEvent() {
  waitForMarkerEvent();
  if (queueCount) {
    // AMD GPUs have no trouble waiting for a finish without a CPU busy wait.  So, instead of markers and events, simply run finish every now and then.
    if (isAmdGpu(context->deviceId())) {
      finish();
    }
    // Enqueue a marker for nVidia GPUs
    else {
      clEnqueueMarkerWithWaitList(get(), 0, NULL, &markerEvent);
      markerQueued = true;
      queueCount = 0;
    }
  }
}

void Queue::waitForMarkerEvent() {
  if (!markerQueued) return;
  // By default, nVidia finish causes a CPU busy wait.  Instead, sleep for a while.  Since we know how many items are enqueued after the marker we can make an
  // educated guess of how long to sleep to keep CPU overhead low.
  while (getEventInfo(markerEvent) != CL_COMPLETE) {
#if defined(__CYGWIN__)
    sleep(1);                                  // 1 second.  A very steep overhead as 500 iterations won't take that long.
#else
    usleep(1 + queueCount * squareTime / 10);  // There are 4 kernels per squaring.  Don't overestimate sleep time.  Divide by 10 instead of 4.
#endif
  }
  markerQueued = false;
}

void Queue::setSquareTime(int time) {
  if (time < 30) time = 30;           // Assume a minimum square time of 30us
  if (time > 3000) time = 3000;       // Assume a maximum square time of 3000us
  squareTime = time;
}

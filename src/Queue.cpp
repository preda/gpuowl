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
  context{&context}
{}

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
}

void Queue::readSync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo) {
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
  ::finish(get());
  events.synced();
}

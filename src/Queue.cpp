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

Queue::Queue(const Args& args, const Context& context) :
  QueueHolder{makeQueue(context.deviceId(), context.get())},
  context{&context}
{}

void Queue::writeTE(cl_mem buf, u64 size, const void* data, TimeInfo* tInfo) {
  events.emplace_back(::write(get(), {}, true, buf, size, data), tInfo);
  events.synced();
}

void Queue::fillBufTE(cl_mem buf, u32 patSize, const void* pattern, u64 size, TimeInfo* tInfo) {
  events.emplace_back(::fillBuf(get(), {}, buf, pattern, patSize, size), tInfo);
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

void Queue::readSync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo) {
  events.emplace_back(read(get(), {}, true, buf, size, out), tInfo);
  events.synced();
}

void Queue::readAsync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo) {
  events.emplace_back(read(get(), {}, false, buf, size, out), tInfo);
}

void Queue::copyBuf(cl_mem src, cl_mem dst, u32 size, TimeInfo* tInfo) {
  events.emplace_back(::copyBuf(get(), {}, src, dst, size), tInfo);
}

void Queue::run(cl_kernel kernel, size_t groupSize, size_t workSize, TimeInfo* tInfo) {
#if 0
  if (n1 >= 100) {
    cl_event e = events->at(10).get();
    guard.unlock();
    Timer t;
    waitForEvents({e});
    std::tie(events, guard) = access();
    // log("Were %u, slept %.2fms, remain %u\n", n1, t.at() * 1000, events->nActive());
  }
#endif

  events.emplace_back(::run(get(), kernel, groupSize, workSize, {}, tInfo->name), tInfo);
}

void Queue::finish() {
  ::finish(get());
  events.synced();
}

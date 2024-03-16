// Copyright (C) Mihai Preda

#include "Queue.h"
#include "Args.h"
#include "TimeInfo.h"
#include "timeutil.h"

#include <cassert>

void Queue::synced() {
  // log("synced %u %u\n", u32(events.size()), u32(pendingWrite.size()));
  clearCompleted();
  assert(events.empty());
  events.clear();
}

void Queue::clearCompleted() {
  while (!events.empty() && events.front().isCompleted()) { events.pop_front(); }
}

Queue::Queue(const Args& args, cl_queue q, bool cudaYield) :
  QueueHolder{q},
  cudaYield{cudaYield}
{}

QueuePtr Queue::make(const Args& args, const Context& context, bool cudaYield) {
  return make_shared<Queue>(args, makeQueue(context.deviceId(), context.get()), cudaYield);
}

vector<cl_event> Queue::inOrder() {
  [[maybe_unused]] u32 n1 = nActive();

#if 0
  if (n1 >= 200) { ::waitForEvents({events.at(50).get()}); }
    // Timer t;
    // log("Were %u, slept %.2fms, remain %u\n", n1, t.at() * 1000, nActive());
#endif

  return events.empty() ? vector<cl_event>{} : vector<cl_event>{events.back().get()};
}

void Queue::writeTE(cl_mem buf, u64 size, const void* data, TimeInfo* tInfo) {
  events.emplace_back(::write(get(), inOrder(), true, buf, size, data), tInfo);
  synced();
}

void Queue::fillBufTE(cl_mem buf, u32 patSize, const void* pattern, u64 size, TimeInfo* tInfo) {
  events.emplace_back(::fillBuf(get(), inOrder(), buf, pattern, patSize, size), tInfo);
}

void Queue::readSync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo) {
  events.emplace_back(read(get(), inOrder(), true, buf, size, out), tInfo);
  synced();
}

void Queue::readAsync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo) {
  events.emplace_back(read(get(), inOrder(), false, buf, size, out), tInfo);
}

void Queue::copyBuf(cl_mem src, cl_mem dst, u32 size, TimeInfo* tInfo) {
  events.emplace_back(::copyBuf(get(), inOrder(), src, dst, size), tInfo);
}

void Queue::run(cl_kernel kernel, size_t groupSize, size_t workSize, TimeInfo* tInfo) {
  events.emplace_back(::run(get(), kernel, groupSize, workSize, inOrder(), tInfo->name), tInfo);
}

void Queue::flush() { ::flush(get()); }

void Queue::finish() {
  if (cudaYield) {
    flush();
    while (nActive()) { Timer::usleep(1000); }
  }

  ::finish(get());
  synced();
}

#if 0
void Queue::write(cl_mem buf, vector<i32>&& vect, TimeInfo* tInfo) {
  pendingWrite.push_back(std::move(vect));
  auto& v = pendingWrite.back();
  events.emplace_back(Event{::write(get(), inOrder(), false, buf, v.size() * sizeof(i32), v.data())}, tInfo);
}
#endif

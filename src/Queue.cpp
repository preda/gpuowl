// Copyright (C) Mihai Preda

#include "Queue.h"
#include "Args.h"
#include "TimeInfo.h"

void Queue::synced() {
  // log("synced %u %u\n", u32(events.size()), u32(pendingWrite.size()));

  events.clear();
  // pendingWrite.clear();
}

Queue::Queue(const Args& args, cl_queue q, bool cudaYield) :
  QueueHolder{q},
  cudaYield{cudaYield}
{}

QueuePtr Queue::make(const Args& args, const Context& context, bool cudaYield) {
  return make_shared<Queue>(args, makeQueue(context.deviceId(), context.get()), cudaYield);
}

vector<cl_event> Queue::inOrder() const {
  return events.empty() ? vector<cl_event>{} : vector<cl_event>{events.back().get()};
}

void Queue::readSync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo) {
  events.emplace_back(read(get(), inOrder(), true, buf, size, out), tInfo);
  synced();
}

void Queue::readAsync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo) {
  events.emplace_back(read(get(), inOrder(), false, buf, size, out), tInfo);
}

#if 0
void Queue::write(cl_mem buf, vector<i32>&& vect, TimeInfo* tInfo) {
  pendingWrite.push_back(std::move(vect));
  auto& v = pendingWrite.back();
  events.emplace_back(Event{::write(get(), inOrder(), false, buf, v.size() * sizeof(i32), v.data())}, tInfo);
}
#endif

void Queue::copyBuf(cl_mem src, cl_mem dst, u32 size, TimeInfo* tInfo) {
  events.emplace_back(::copyBuf(get(), inOrder(), src, dst, size), tInfo);
}

void Queue::run(cl_kernel kernel, size_t groupSize, size_t workSize, TimeInfo* tInfo) {
  events.emplace_back(::run(get(), kernel, groupSize, workSize, inOrder(), tInfo->name), tInfo);
}

bool Queue::allEventsCompleted() { return events.empty() || events.back().isDone(); }

void Queue::flush() { ::flush(get()); }

void Queue::finish() {
  if (cudaYield) {
    flush();
    while (!allEventsCompleted()) {
#if defined(__CYGWIN__)
      sleep(1);
#else
      usleep(500);
#endif
    }
  }

  ::finish(get());
  synced();
}

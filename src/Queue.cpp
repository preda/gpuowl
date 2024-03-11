// Copyright (C) Mihai Preda

#include "Queue.h"
// #include "Kernel.h"
#include "Args.h"
#include "TimeInfo.h"

void Queue::synced() {
  // log("synced %u\n", u32(pendingWrite.size()));
  flushPos.reset();
  pendingWrite.clear();
}

Queue::Queue(const Args& args, cl_queue q, bool profile, bool cudaYield) :
  QueueHolder{q},
  profile{profile},
  cudaYield{cudaYield},
  flushPos{args}
{}

QueuePtr Queue::make(const Args& args, const Context& context, bool profile, bool cudaYield) {
  return make_shared<Queue>(args, makeQueue(context.deviceId(), context.get(), profile), profile, cudaYield);
}

void Queue::readSync(cl_mem buf, u32 size, void* out) {
  ::read(get(), true, buf, size, out);
  synced();
}

void Queue::write(cl_mem buf, u32 size, const void* data) {
  ::write(get(), true, buf, size, data);
}

void Queue::write(cl_mem buf, vector<i32>&& vect) {
  pendingWrite.push_back(std::move(vect));
  auto& v = pendingWrite.back();
  ::write(get(), false, buf, v.size() * sizeof(i32), v.data());
}

void Queue::run(cl_kernel kernel, size_t groupSize, size_t workSize, TimeInfo* tInfo) {
  Event event{::run(get(), kernel, groupSize, workSize, tInfo->name, true)};
  events.emplace_back(std::move(event), tInfo);

  if (flushPos.inc()) { flush(); }
}

bool Queue::allEventsCompleted() { return events.empty() || events.back().first.isComplete(); }

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

  for (auto& [event, tinfo] : events) { tinfo->add(event.times()); }
  events.clear();
  synced();
}

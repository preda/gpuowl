// Copyright (C) Mihai Preda

#include "Queue.h"
#include "Args.h"
#include "TimeInfo.h"

#include <atomic>
#include <cassert>

static thread_local int threadId = -1;
static atomic<int> nThread = 0;

// Because Event is unique_ptr<cl_event>, it can't be copied.
// As a consequence, Events (which is a container of Event) can't be copied either.
// Except the empty Events which can be copied as there's not Event copy involved in that case.
Events::Events(const Events& oth) { assert(oth.empty()); }

void Events::clearCompleted() { while (!empty() && front().isCompleted()) { pop_front(); } }

void Events::synced() {
  clearCompleted();
  assert(empty());
}

u32 Events::nActive() {
  clearCompleted();
  return size();
}

vector<cl_event> Events::inOrder() {
  [[maybe_unused]] u32 n1 = nActive();
  return empty() ? vector<cl_event>{} : vector<cl_event>{back().get()};

#if 0
  if (n1 >= 200) { ::waitForEvents({events.at(50).get()}); }
    // Timer t;
    // log("Were %u, slept %.2fms, remain %u\n", n1, t.at() * 1000, nActive());
#endif
}

// static method
int Queue::tid() {
  assert(threadId >= 0);
  return threadId;
}

// static method
int Queue::registerThread() {
  assert(threadId == -1);
  threadId = nThread++;
  return threadId;
}

Queue::Queue(const Args& args, const Context& context) :
  QueueHolder{makeQueue(context.deviceId(), context.get())},
  context{&context}
{}

std::pair<Events*, unique_lock<mutex> > Queue::access() {
  assert(threadId >= 0);
  unique_lock guard{mut};
  if (threadId >= int(queues.size())) { queues.resize(threadId + 1); }
  return {&queues.at(threadId), std::move(guard)};
}

void Queue::writeTE(cl_mem buf, u64 size, const void* data, TimeInfo* tInfo) {
  auto [events, guard] = access();
  events->emplace_back(::write(get(), events->inOrder(), true, buf, size, data), tInfo);
  events->synced();
}

void Queue::fillBufTE(cl_mem buf, u32 patSize, const void* pattern, u64 size, TimeInfo* tInfo) {
  auto [events, guard] = access();
  events->emplace_back(::fillBuf(get(), events->inOrder(), buf, pattern, patSize, size), tInfo);
}

void Queue::readSync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo) {
  auto [events, guard] = access();
  events->emplace_back(read(get(), events->inOrder(), true, buf, size, out), tInfo);
  events->synced();
}

void Queue::readAsync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo) {
  auto [events, guard] = access();
  events->emplace_back(read(get(), events->inOrder(), false, buf, size, out), tInfo);
}

void Queue::copyBuf(cl_mem src, cl_mem dst, u32 size, TimeInfo* tInfo) {
  auto [events, guard] = access();
  events->emplace_back(::copyBuf(get(), events->inOrder(), src, dst, size), tInfo);
}

void Queue::run(cl_kernel kernel, size_t groupSize, size_t workSize, TimeInfo* tInfo) {
  auto [events, guard] = access();
  events->emplace_back(::run(get(), kernel, groupSize, workSize, events->inOrder(), tInfo->name), tInfo);
}

void Queue::finish() {
  auto [events, guard] = access();
  auto wait = events->inOrder();
  guard.unlock();
  waitForEvents(std::move(wait));
}

#if 0
void Queue::write(cl_mem buf, vector<i32>&& vect, TimeInfo* tInfo) {
  pendingWrite.push_back(std::move(vect));
  auto& v = pendingWrite.back();
  events.emplace_back(Event{::write(get(), inOrder(), false, buf, v.size() * sizeof(i32), v.data())}, tInfo);
}
#endif

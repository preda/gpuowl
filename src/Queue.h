// Copyright (C) Mihai Preda

#pragma once

#include "common.h"
#include "clwrap.h"
#include "Context.h"
#include "log.h"
#include "Args.h"

#include <memory>
#include <vector>
#include <unistd.h>

class Args;

template<typename T> class ConstBuffer;
template<typename T> class Buffer;

struct TimeInfo;

class Event {
  mutable bool isFinalized{false};

public:
  EventHolder event;
  TimeInfo *tInfo;

  Event(EventHolder&& e, TimeInfo *tInfo);
  Event(Event&& oth) = default;
  ~Event();

  cl_event get() const { return event.get(); }
  bool isDone() const;
};

using QueuePtr = std::shared_ptr<class Queue>;

class Queue : public QueueHolder {
  std::vector<Event> events;

  bool cudaYield{};
  // vector<vector<i32>> pendingWrite;

  void synced();

  vector<cl_event> inOrder() const;

  void writeTE(cl_mem buf, u64 size, const void* data, TimeInfo *tInfo) {
    events.emplace_back(::write(get(), inOrder(), true, buf, size, data), tInfo);
    synced();
  }

  void fillBufTE(cl_mem buf, u32 patSize, const void* pattern, u64 size, TimeInfo* tInfo) {
    events.emplace_back(::fillBuf(get(), inOrder(), buf, pattern, patSize, size), tInfo);
  }

public:
  static QueuePtr make(const Args& args, const Context& context, bool cudaYield);

  Queue(const Args& args, cl_queue q, bool cudaYield);

  void run(cl_kernel kernel, size_t groupSize, size_t workSize, TimeInfo* tInfo);

  void readSync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo);
  void readAsync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo);

  template<typename T>
  void write(cl_mem buf, const vector<T>& v, TimeInfo* tInfo) {
    writeTE(buf, v.size() * sizeof(T), v.data(), tInfo);
  }

  // void write(cl_mem buf, vector<i32>&& vect, TimeInfo* tInfo);

  template<typename T>
  void fillBuf(cl_mem buf, T pattern, u32 size, TimeInfo* tInfo) {
    fillBufTE(buf, sizeof(T), &pattern, size, tInfo);
    // events.emplace_back(::fillBuf(get(), inOrder(), buf, &pattern, sizeof(T), size), tInfo);
  }

  void copyBuf(cl_mem src, cl_mem dst, u32 size, TimeInfo* tInfo);

  bool allEventsCompleted();

  void flush();
  
  void finish();

  using Profile = std::vector<TimeInfo>;

  Profile getProfile();

  void clearProfile();
};

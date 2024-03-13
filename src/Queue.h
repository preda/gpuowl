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
#include <array>

class Args;

template<typename T> class ConstBuffer;
template<typename T> class Buffer;

class Event : public EventHolder {
public:
  // double secs() { return getEventNanos(this->get()) * 1e-9f; }
  bool isComplete() { return getEventInfo(this->get()) == CL_COMPLETE; }
  std::array<i64, 3> times() { return getEventNanos(get()); }
};

class FlushPolicy {
private:
  const u32 step;
  u32 pos;
  u32 posFlush;

public:
  FlushPolicy(const Args& args) :
    step{args.flush}
  {
    reset();
  }

  void reset() {
    pos = 0;
    posFlush = step;
  }

  u32 get() const { return pos; }

  // return true if should flush now
  bool inc() {
    ++pos;
    if (!step || pos != posFlush) { return false; }
    // posNextFlush += step;
    return true;
  }
};

using QueuePtr = std::shared_ptr<class Queue>;

struct TimeInfo;

class Queue : public QueueHolder {
  // using TimeMap = std::map<std::string, TimeInfo>;
  // TimeMap timeMap;

  std::vector<std::pair<Event, TimeInfo*>> events;

  bool cudaYield{};
  FlushPolicy flushPos;
  // vector<vector<i32>> pendingWrite;

  static constexpr const u32 FLUSH_FACTOR = 4;

  void synced();

  vector<cl_event> inOrder() const;

public:
  static QueuePtr make(const Args& args, const Context& context, bool cudaYield);

  Queue(const Args& args, cl_queue q, bool cudaYield);

  void run(cl_kernel kernel, size_t groupSize, size_t workSize, TimeInfo* tInfo);

  void readSync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo);
  void readAsync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo);

  template<typename T>
  void write(cl_mem buf, const vector<T>& v, TimeInfo* tInfo) {
    events.emplace_back(Event{::write(get(), inOrder(), true, buf, v.size() * sizeof(T), v.data())}, tInfo);
    synced();
  }

  // void write(cl_mem buf, vector<i32>&& vect, TimeInfo* tInfo);

  template<typename T>
  void fillBuf(cl_mem buf, T pattern, u32 size, TimeInfo* tInfo) {
    events.emplace_back(Event{::fillBuf(get(), inOrder(), buf, &pattern, sizeof(T), size)}, tInfo);
  }

  void copyBuf(cl_mem src, cl_mem dst, u32 size, TimeInfo* tInfo);

  bool allEventsCompleted();

  void flush();
  
  void finish();

  using Profile = std::vector<TimeInfo>;

  Profile getProfile();

  void clearProfile();
};

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

// class Kernel;
class Args;

template<typename T> class ConstBuffer;
template<typename T> class Buffer;

/*
struct TimeInfo {
  double total{};
  u32 n{};

  bool operator<(const TimeInfo& rhs) const { return total > rhs.total; }
  void add(float deltaTime, u32 deltaN = 1) { total += deltaTime; n += deltaN; }
  void clear() { total = 0; n = 0; }
};
*/

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

  bool profile{};
  bool cudaYield{};
  FlushPolicy flushPos;
  vector<vector<i32>> pendingWrite;

  static constexpr const u32 FLUSH_FACTOR = 4;

  void synced();

public:
  static QueuePtr make(const Args& args, const Context& context, bool profile, bool cudaYield);

  Queue(const Args& args, cl_queue q, bool profile, bool cudaYield);

  void run(cl_kernel kernel, size_t groupSize, size_t workSize, TimeInfo* tInfo);

  void readSync(cl_mem buf, u32 size, void* out);

  void write(cl_mem buf, u32 size, const void* data);

  void write(cl_mem buf, vector<i32>&& vect);


  bool allEventsCompleted();

  void flush();
  
  void finish();

  using Profile = std::vector<TimeInfo>;

  Profile getProfile();

  void clearProfile();
};

// Copyright Mihai Preda.

#pragma once

#include "common.h"
#include "clwrap.h"
#include "Context.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>
#include <assert.h>

template<typename T> class ConstBuffer;
template<typename T> class Buffer;

struct TimeInfo {
  double total{};
  u32 n{};

  bool operator<(const TimeInfo& rhs) const { return total > rhs.total; }
  void add(float deltaTime, u32 deltaN = 1) { total += deltaTime; n += deltaN; }
  void clear() { total = 0; n = 0; }
};

class Event : public EventHolder {
public:
  double secs() { return getEventNanos(this->get()) * 1e-9f; }
  bool isComplete() { return getEventInfo(this->get()) == CL_COMPLETE; }
};

class FlushPolicy {
private:
  u32 pos;
  u32 posNextFlush;
  u32 step;

public:
  FlushPolicy() { reset(); }

  void reset() {
    pos = 0;
    posNextFlush = 1;
    step = 1;
  }

  u32 get() const { return pos; }

  // return true if should flash now
  bool inc() {

#if 1
    return false;
#else
    ++pos;
    return pos == 4;
#endif

    // return false;
    /*
    assert(pos <= posNextFlush);
    if (pos >= posNextFlush) {
      step = std::min(step * 3, 100u);
      posNextFlush += step;
      return true;
    }
    return false;
    */
  }
};

using QueuePtr = std::shared_ptr<class Queue>;

class Queue : public QueueHolder {
  using TimeMap = std::map<std::string, TimeInfo>;
  TimeMap timeMap;
  std::vector<std::pair<Event, TimeMap::iterator>> events;
  bool profile{};
  bool cudaYield{};
  FlushPolicy flushPos;
  vector<vector<i32>> pendingWrite;

  static constexpr const u32 FLUSH_FACTOR = 4;

  void synced() {
    // log("synced %u\n", u32(pendingWrite.size()));
    flushPos.reset();
    pendingWrite.clear();
  }

public:
  Queue(cl_queue q, bool profile, bool cudaYield) : QueueHolder{q}, profile{profile}, cudaYield{cudaYield} {}  
  static QueuePtr make(const Context& context, bool profile, bool cudaYield) { return make_shared<Queue>(makeQueue(context.deviceId(), context.get(), profile), profile, cudaYield); }
  
  void readSync(cl_mem buf, u32 size, void* out) {
    ::read(get(), true, buf, size, out);
    synced();
  }

  void write(cl_mem buf, u32 size, const void* data) {
    ::write(get(), true, buf, size, data);
  }

  void write(cl_mem buf, vector<i32>&& vect) {
    pendingWrite.push_back(std::move(vect));
    auto& v = pendingWrite.back();
    ::write(get(), false, buf, v.size() * sizeof(i32), v.data());
  }

  void run(cl_kernel kernel, size_t groupSize, size_t workSize, const string &name) {
    if (!profile && !cudaYield) {
      ::run(get(), kernel, groupSize, workSize, name, false);
    } else {
      Event event{::run(get(), kernel, groupSize, workSize, name, true)};
      auto it = profile ? timeMap.insert({name, TimeInfo{}}).first : timeMap.end();

      if (cudaYield && !profile && !events.empty()) {
        assert(events.size() == 1);
        events.pop_back();
      }

      events.emplace_back(std::move(event), it);
    }
    if (flushPos.inc()) {
      // log("flush at %u\n", flushPos.get());
      flush();
    }
  }

  bool allEventsCompleted() { return events.empty() || events.back().first.isComplete(); }

  void flush() { ::flush(get()); }
  
  void finish() {
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
    
    if (profile) { for (auto& [event, it] : events) { it->second.add(event.secs()); } }
    events.clear();
    synced();
  }

  using Profile = std::vector<std::pair<TimeInfo, std::string>>;
  Profile getProfile() {
    Profile p;
    for (auto& [name, info] : timeMap) { p.emplace_back(info, name); }
    std::sort(p.begin(), p.end());
    return p;
  }

  void clearProfile() {
    events.clear();
    timeMap.clear();
  }

  /*
  template<typename T> void zero(Buffer<T>& buf, size_t sizeOrFull = 0) {
    auto size = sizeOrFull ? sizeOrFull : buf.size;
    assert(size <= buf.size);
    T zero = 0;
    fillBuf(get(), buf.get(), &zero, sizeof(T), size * sizeof(T));
  }
  */
};

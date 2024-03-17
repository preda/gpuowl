// Copyright (C) Mihai Preda

#pragma once

#include "common.h"
#include "clwrap.h"
#include "Context.h"
#include "Args.h"
#include "Event.h"

#include <deque>
#include <vector>
#include <unistd.h>

class Args;
class TimeInfo;

// template<typename T> class Buffer;

class Queue : public QueueHolder {
  std::deque<Event> events;

  void synced();
  void clearCompleted();

  u32 nActive() {
    clearCompleted();
    return events.size();
  }

  vector<cl_event> inOrder();

  void writeTE(cl_mem buf, u64 size, const void* data, TimeInfo *tInfo);
  void fillBufTE(cl_mem buf, u32 patSize, const void* pattern, u64 size, TimeInfo* tInfo);

public:
  const Context* context;

  Queue(const Args& args, const Context& context);

  void run(cl_kernel kernel, size_t groupSize, size_t workSize, TimeInfo* tInfo);

  void readSync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo);
  void readAsync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo);

  template<typename T>
  void write(cl_mem buf, const vector<T>& v, TimeInfo* tInfo) {
    writeTE(buf, v.size() * sizeof(T), v.data(), tInfo);
  }

  template<typename T>
  void fillBuf(cl_mem buf, T pattern, u32 size, TimeInfo* tInfo) {
    fillBufTE(buf, sizeof(T), &pattern, size, tInfo);
  }

  void copyBuf(cl_mem src, cl_mem dst, u32 size, TimeInfo* tInfo);

  void flush();
  
  void finish();
};
